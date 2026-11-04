import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time


# 从HuggingFace模型加载权重
def load_weights_from_hf(model:nn.Module, model_path:str, device:torch.device, dtype:torch.dtype):
    from transformers import AutoModelForCausalLM
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, dtype = dtype, device_map = "cpu")
    hf_state_dict = hf_model.state_dict()

    result = model.load_state_dict(hf_state_dict, strict = False)
    if result.missing_keys:
        print(f"Warning: Missing keys when loading weights: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: Unexpected keys when loading weights: {result.unexpected_keys}")
    model.to(device=device, dtype=dtype)
    

def _resolve_model_path(model_path: str) -> str:
    import os
    if os.path.isdir(model_path):
        return model_path
    from huggingface_hub import snapshot_download
    print(f"Downloading model from Huggingface: {model_path}...")
    local_dir = snapshot_download(model_path, allow_patterns = ["*.safetensors", "*.bin"])
    return local_dir


# 从safetensors文件加载权重
# 大模型推理引擎通常会提供这个功能
def load_weights_from_safetensors(model:nn.Module, model_path:str, device:torch.device, dtype:torch.dtype):
    import glob
    import safetensors
    # model_path 可以是一个model目录，也可以是huggingface模型名
    # 如果是目录则加载其中的safetensors文件，如果是huggingface模型名，从huggingface下载权重文件，得到目录路径加载
    local_dir = _resolve_model_path(model_path)
    files = glob.glob(f"{local_dir}/*.safetensors")
    if not files:
        raise FileNotFoundError(f"No safetensors files found in {local_dir}")

    # 读取所有权重文件到一个大的 state_dict 中
    state_dict = {}
    for file in files:
        with safetensors.safe_open(file, framework = "pt", device = "cpu") as f:
            for name in f.keys():
                state_dict[name] = f.get_tensor(name)
    
    # 数据类型转换, device转换
    state_dict = {k: v.to(device = device, dtype = dtype) for k, v in state_dict.items()}

    # 加载权重到模型
    result = model.load_state_dict(state_dict, strict = False)
    if result.missing_keys:
        print(f"Warning: Missing keys when loading weights: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"Warning: Unexpected keys when loading weights: {result.unexpected_keys}")
    model.to(device=device)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) # gamma
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, hidden_size)
        # 1.计算RMS: sqrt(mean(x^2))
        rms = torch.sqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps)
        # 归一化并缩放
        return self.weight * (x / rms)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 1000000.0):
        super().__init__()
        # 1.计算head_dim维度上的旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))  # 这样直接计算得到的是fp32,和qk相乘会提升qk精度类型
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, position_ids):
        # 2.乘以位置ID -> 每个位置的旋转角度
        freqs = torch.outer(position_ids[0].float(), self.inv_freq) # (seq_len, head_dim/2)
        # 3.计算 cos/sin
        emb = torch.cat((freqs, freqs), dim = -1) # (seq_len, head_dim)
        return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0) # (1, seq_len, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin):
    # 4&5：旋转矩阵乘法，应用到Q和K
    cos = cos.unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0)

    def rotate_half(x):
        x1, x2 = x.chunk(2, dim = -1)
        return torch.cat((-x2, x1), dim = -1)

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


def repeat_kv(x, num_groups):
        """ 将KV头复制 num_groups 次,以匹配Q的头数"""
        if num_groups == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, num_groups, s, d).reshape(b, h * num_groups, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5 # 1/根号dk

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias = False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias = False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias = False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias = False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(self, 
                hidden_states: torch.Tensor, # (batch, seq_len, hidden_size)
                attention_mask: torch.Tensor, # (1, 1, seq_len, seq_len)
                position_embedding: tuple[torch.Tensor, torch.Tensor], # cos sin  (1, seq_len, head_dim)
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2) # (B, num_heads, S, head_dim)
        k = self.k_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embedding
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale + attention_mask
        attn_probs = F.softmax(attn_scores, dim = -1, dtype = torch.float32).to(v.dtype) # (B, num_heads, S, S)
        attn_output = torch.matmul(attn_probs, v) # (B, num_heads, S, head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, -1) # (B, S, num_heads * head_dim)
        
        return self.o_proj(attn_output)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias = False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias = False)

    def forward(self, hidden_states):
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_id)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden_states, attention_mask, position_embedding):
        # hidden_states: (batch, seq_len, hidden_size)
        # 1.自注意力
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, position_embedding)
        hidden_states = residual + hidden_states

        # 2.MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_parameters["rope_theta"])
    
    def forward(self, input_ids:torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape

        # Embedding : 每个token id -> tensor
        hidden_states = self.embed_tokens(input_ids)

        # 计算位置id (S) -> (1, S) -> (B, S)
        position_ids = torch.arange(S, device = input_ids.device).unsqueeze(0).expand(B, -1)
        
        # 计算RoPE cos/sin
        position_embedding = self.rotary_emb(position_ids)
        
        # 因果掩码：上三角 -inf
        casual_mask = torch.full((S, S), float("-inf"), device = input_ids.device, dtype = hidden_states.dtype)
        casual_mask = torch.triu(casual_mask, diagonal = 1)
        casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            hidden_states = layer(hidden_states, casual_mask, position_embedding)

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias = False)

    def forward(self, input_ids) -> torch.Tensor:
        hidden_states = self.model(input_ids)
        return self.lm_head(hidden_states) # (batch, seq_len, hidden_size) -> (batch, seq_len, vocab_size)


def generate(model, input_ids, max_new_tokens, temperature, top_k, eos_token_id) -> torch.Tensor:
    # 自回归
    generated = input_ids.clone()
    stats = {"step_times": [], "input_len": input_ids.shape[1]}
    for step in range(max_new_tokens):
        t0 = time.perf_counter()

        # 没有 KV cache: 每步送入完整序列
        logits = model(generated) # (batch, seq_len, vocab_size)
        dt = time.perf_counter() - t0
        stats["step_times"].append(dt)

        next_logits = logits[:, -1, :]
        # 采样
        if temperature == 0:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            next_logits = next_logits / temperature
            if top_k > 0:
                topk_vals = torch.topk(next_logits, min(top_k, next_logits.shape[-1])).values # (batch, 1, k)
                next_logits = next_logits.masked_fill(next_logits < topk_vals[..., -1:], float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples = 1)
        generated = torch.cat([generated, next_token], dim = -1)
        if next_token.item() == eos_token_id:
            break
    return generated, stats


def main():
    parser = argparse.ArgumentParser(description="手搓qwen3推理")
    parser.add_argument("--model", type=str, help="HuggingFace 模型名或本地路径, e.g. Qwen/Qwen3-0.6B", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--prompt", type=str, default="Who are you?")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default=None,
                        help="指定设备, e.g. 'cuda:1' (默认: 自动选择最空闲的 GPU)")

    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        best = free.index(max(free))
        device = torch.device(f"cuda:{best}")
    else:
        device = torch.device("cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    print(f"\nDevice: {device}, dtype: {dtype}")

    # 读取config
    from transformers import AutoTokenizer, AutoConfig
    print(f"\n[1/4] 读取模型配置...")
    config = AutoConfig.from_pretrained(args.model)
    print(f"  Model: {args.model}")
    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}, "
          f"Heads: {config.num_attention_heads}, KV Heads: {config.num_key_value_heads}")

    # 创建模型
    print(f"\n[2/4] 创建模型结构...")
    model = Qwen3ForCausalLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {param_count:,} ({param_count / 1e9:.2f}B)")

    # 加载权重
    print(f"\n[3/4] 从 HuggingFace 加载权重...")
    load_weights_from_hf(model, args.model, device, dtype)
    model.eval()

    # 推理生成
    print(f"\n[4/4] 生成文本...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)

    sampling = "greedy" if args.temperature == 0 else f"T={args.temperature}"
    print(f"  Prompt: {args.prompt}")
    print(f"  Input tokens: {input_ids.shape[1]}")
    print(f"  Sampling: {sampling}, max_tokens: {args.max_tokens}")

    t0 = time.perf_counter()
    output_ids, stats = generate(
        model, input_ids,
        max_new_tokens = args.max_tokens,
        temperature = args.temperature,
        top_k = args.top_k,
        eos_token_id = tokenizer.eos_token_id,
    )
    t_total = time.perf_counter() - t0

    # output_ids -> [input_ids, ......]
    new_tokens = output_ids[0][input_ids.shape[1]:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens = True)
    print("Generated:")
    print("=" * 60)
    print(output_text)
    print("=" * 60)

# --- 统计 ---
    n = len(new_tokens)
    step_times = stats["step_times"]
    print(f"\n统计:")
    print(f"  生成 tokens: {n}")
    print(f"  总耗时: {t_total:.2f}s")
    if n > 0:
        print(f"  平均每步: {sum(step_times) / len(step_times) * 1000:.1f} ms")
        print(f"  首步 (prefill): {step_times[0] * 1000:.1f} ms")
        if len(step_times) > 1:
            print(f"  末步: {step_times[-1] * 1000:.1f} ms")

    print(f"\n注意: 没有 KV cache, 每步重新计算整个序列")
    print(f"  Step 1 处理 {stats['input_len']} tokens")
    print(f"  Step {n} 处理 {stats['input_len'] + n - 1} tokens")
    print(f"  KV cache (Lesson 4) 可将 decode 步骤从 O(seq_len) 降为 O(1)")

if __name__ == "__main__":
    main()