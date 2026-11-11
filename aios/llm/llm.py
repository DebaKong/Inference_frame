from __future__ import annotations

import os
from typing import List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer

from ..models import ModelConfig, create_model, load_weights
from ..engine import Sampler, DynamicKVCache
from ..engine.engine import Engine
from ..kvcache import MHAKVCache, KVCacheLayout
from ..scheduler import CacheManager
from ..scheduler.table import TableManager
from ..scheduler.scheduler import Scheduler 
from ..core import SamplingParams, Req


def _resolve_model_path(model_path: str) -> str:
    """Resolve a HuggingFace hub ID to a local path if needed."""
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


class LLM:
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        self.device = torch.device(kwargs.get("device", "cuda"))
        self.dtype = dtype

        model_path = _resolve_model_path(model_path)
        hf_config = AutoConfig.from_pretrained(model_path)
        config = ModelConfig.from_hf(hf_config)
        self._num_layers = config.num_layers

        with torch.device("meta"):
            self.model = create_model(model_path, config)

        load_weights(self.model, model_path, self.device, self.dtype)

        # Move rotary embedding cache to device

        self.model.model._rotary_emb.set_device(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Paged KV Cache Memory Pool (mini-sglang style: use remaining GPU memory)
        self.num_pages = self._determine_num_pages(config, kwargs.get("memory_ratio", 0.8))
        self.mha_kv_cache = MHAKVCache(
            num_kv_heads=config.num_kv_heads,
            num_layers=config.num_layers,
            head_dim=config.head_dim,
            num_pages=self.num_pages,
            dtype=self.dtype,
            kv_layout=KVCacheLayout.LayerFirst,
            device=self.device,
            page_size=1
        )
        self.cache_manager = CacheManager(self.device, self.num_pages)

    def _determine_num_pages(self, config: ModelConfig, memory_ratio: float) -> int:
        """Determine num_pages from remaining GPU memory (mini-sglang style)."""
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        free_memory = torch.cuda.mem_get_info(self.device)[0]
        # KV cache per page: 2 (K+V) * head_dim * num_kv_heads * page_size * dtype_bytes * num_layers
        cache_per_page = (
            2 * config.head_dim * config.num_kv_heads * 1 * self.dtype.itemsize * config.num_layers
        )
        available_memory = int(memory_ratio * free_memory)
        num_pages = available_memory // cache_per_page
        assert num_pages > 1, f"Not enough GPU memory for KV cache (free={free_memory}, per_page={cache_per_page})"
        return num_pages

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams] | None = None,
        use_kv_cache: bool = True,
        use_paged_kv_cache: bool = False,
        use_static_batch: bool = False,
    ) -> List[dict]:
        # 归一化 sampling_params:统一为「每个 prompt 对应一个 SamplingParams」的列表
        if sampling_params is None:
            params_list = [SamplingParams()] * len(prompts)

        if use_static_batch:
            return self._generate_static_batch_paged(prompts, sampling_params)

        # Normalize to per-request sampling params list
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            # List[SamplingParams],长度需与 prompts 一致
            assert len(sampling_params) == len(prompts), (
                f"sampling_params length ({len(sampling_params)}) != prompts length ({len(prompts)})"
            )
            params_list = sampling_params

        results = []
        for req_id, (prompt, sp) in enumerate(zip(prompts, params_list)):
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]

                text = self.tokenizer.apply_chat_template(
                    messages, tokenize = False,
                    add_generation_prompt = True, enable_thinking = False
                )
                input_ids = self.tokenizer.encode(text, return_tensors = "pt").to(self.device)
            else:
                input_ids = torch.tensor([prompt], dtype = torch.long, device = self.device)

            sampler = Sampler(sp)
            generated = input_ids.clone()

            kv_cache = None
            req: Req | None =  None
            
            if use_paged_kv_cache:
                block_table = self.cache_manager.allocate(input_ids.shape[1])
                req = Req(
                    input_ids=input_ids[0],
                    cached_len=0,
                    output_len=sp.max_tokens,
                    uid=req_id,
                    sampling_params=sp,
                    block_table=block_table,
                )
            elif use_kv_cache:
                kv_cache = DynamicKVCache(self._num_layers)
            
            model_input = input_ids

            for step in range(sp.max_tokens):
                if use_paged_kv_cache:
                    assert req is not None
                    logits = self.model.forward(model_input, paged_kv_cache = self.mha_kv_cache, req=req)
                else:
                    logits = self.model.forward(model_input, kv_cache = kv_cache)
                
                next_logits = logits[:, -1, :]
                next_token = sampler.sample(next_logits)

                generated = torch.cat([generated, next_token], dim = -1)
                if not sp.ignore_eos and next_token.item() == self.tokenizer.eos_token_id:
                    break

                if use_paged_kv_cache:
                    assert req is not None and req.block_table is not None
                    req.complete_one()
                    new_block = self.cache_manager.allocate(1)
                    req.block_table = torch.cat([req.block_table, new_block])
                
                model_input = next_token if (kv_cache is not None or use_paged_kv_cache) else generated

            if req is not None and req.block_table is not None:
                self.cache_manager._free(req.block_table[: req.cached_len])
            
            new_token_ids = generated[0][input_ids.shape[1]:].tolist()
            text = self.tokenizer.decode(new_token_ids, skip_special_tokens = True)
            results.append({"text": text, "token_ids": new_token_ids})

        return results

    @torch.no_grad()
    def _generate_static_batch_paged(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: SamplingParams | List[SamplingParams],
    ) -> List[dict]:
        """Static batch generation with paged KV cache ."""
        # Normalize sampling params
        if isinstance(sampling_params, SamplingParams):
            params_list = [sampling_params] * len(prompts)
        else:
            params_list = sampling_params

        # Tokenize all prompts -> list of 1D CPU tensors
        all_inputs_ids: List[torch.Tensor] = []
        for prompt in prompts:
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                ids = self.tokenizer.encode(text, return_tensors="pt")[0]
            else:
                ids = torch.tensor(prompt)
            all_inputs_ids.append(ids)

        # Create TableManager
        max_running_reqs = len(prompts)
        max_total_len = max(
            len(ids) + sp.max_tokens
            for ids, sp in zip(all_inputs_ids, params_list)
        )
        page_table = torch.zeros((max_running_reqs, max_total_len), dtype=torch.int32, device=self.device)
        table_manager = TableManager(max_running_reqs, page_table)

        # Create Scheduler
        scheduler = Scheduler(
            table_manager=table_manager,
            cache_manager=self.cache_manager,
            eos_token_id=self.tokenizer.eos_token_id,
            device=self.device,
        )

        # Create Engine
        engine = Engine(model=self.model, mha_kv_cache=self.mha_kv_cache)

        # Init requests
        scheduler.init_requests(all_inputs_ids, params_list)

        # Prefill phase
        for scheduled in scheduler.iter_prefill_batches():
            next_tokens = engine.run_batch(scheduled)
            scheduler.process_batch_output(scheduled, next_tokens)

        # Decode loop
        while scheduler.has_active_reqs:
            scheduled = scheduler.schedule_decode_batch()
            if scheduled is None:
                break
            next_tokens = engine.run_batch(scheduled)
            scheduler.process_batch_output(scheduled, next_tokens)

        # Finalize and return results
        return scheduler.finalize_results(self.tokenizer)
