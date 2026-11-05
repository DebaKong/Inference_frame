import argparse
import time
from random import randint, seed
from typing import List

import torch

from aios.core import SamplingParams, Req
from aios.engine import Sampler, DynamicKVCache
from aios.llm import LLM


def run_one_sequence(
    llm: LLM,
    input_ids: torch.Tensor,
    sampling_params: SamplingParams,
    use_kv_cache: bool,
    use_paged_kv_cache: bool,
    sync: bool = True,
) -> dict:
    """Drive a single sequence end-to-end and collect timing metrics.

    Supports three modes:
      - no cache          : recompute full sequence every step
      - dynamic KV cache  : grows with torch.cat (lesson-4)
      - paged KV cache    : pre-allocated paged pool (lesson-6)

    Returns a dict with:
      - ttft:        time from first forward to first generated token (prefill)
      - prefill_time:time of the first forward (prefill) step
      - decode_time: total time of all decode steps (after the first token)
      - decode_times:per-token decode latencies (TPOT/ITL list)
      - total_time:  ttft + decode_time
      - output_tokens: number of generated tokens
    """
    device = llm.device
    sampler = Sampler(sampling_params)
    num_layers = llm._num_layers
    eos = llm.tokenizer.eos_token_id

    def _sync():
        if sync and device.type == "cuda":
            torch.cuda.synchronize()

    # ---- 按模式初始化 KV cache 与 req ----
    kv_cache = None
    req = None
    if use_paged_kv_cache:
        # 复用 LLM 内部的 paged 内存池与页管理器
        block_table = llm.cache_manager.allocate(input_ids.shape[1])
        req = Req(
            input_ids=input_ids[0],
            cached_len=0,
            output_len=sampling_params.max_tokens,
            uid=0,
            sampling_params=sampling_params,
            block_table=block_table,
        )
    elif use_kv_cache:
        kv_cache = DynamicKVCache(num_layers)

    generated = input_ids.clone()

    def _forward(model_input):
        if use_paged_kv_cache:
            assert req is not None
            return llm.model.forward(model_input, paged_kv_cache=llm.mha_kv_cache, req=req)
        return llm.model.forward(model_input, kv_cache=kv_cache)

    # ---- prefill: 一次性前向完整 prompt ----
    _sync()
    t_prefill_start = time.perf_counter()
    logits = _forward(input_ids)
    next_logits = logits[:, -1, :]
    next_token = sampler.sample(next_logits)
    _sync()
    prefill_time = time.perf_counter() - t_prefill_start

    generated = torch.cat([generated, next_token], dim=-1)

    # ---- decode: 逐个 token 生成 ----
    decode_times: List[float] = []
    model_input = next_token if (use_kv_cache or use_paged_kv_cache) else generated

    for _ in range(sampling_params.max_tokens - 1):
        _sync()
        t_step_start = time.perf_counter()
        logits = _forward(model_input)
        next_logits = logits[:, -1, :]
        next_token = sampler.sample(next_logits)
        _sync()
        step_time = time.perf_counter() - t_step_start
        decode_times.append(step_time)

        generated = torch.cat([generated, next_token], dim=-1)
        if not sampling_params.ignore_eos and next_token.item() == eos:
            break

        if use_paged_kv_cache:
            assert req is not None and req.block_table is not None
            req.complete_one()
            new_block = llm.cache_manager.allocate(1)
            req.block_table = torch.cat([req.block_table, new_block])

        model_input = next_token if (use_kv_cache or use_paged_kv_cache) else generated

    # ---- 释放 paged 页 ----
    if req is not None and req.block_table is not None:
        llm.cache_manager._free(req.block_table[: req.cached_len])

    output_tokens = generated.shape[1] - input_ids.shape[1]
    decode_time = sum(decode_times)

    return {
        "ttft": prefill_time,
        "prefill_time": prefill_time,
        "decode_time": decode_time,
        "decode_times": decode_times,
        "total_time": prefill_time + decode_time,
        "output_tokens": output_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="AIOS offline benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--no-kv-cache", action="store_true", help="Disable lesson-4 KV cache path")
    parser.add_argument("--paged-kv-cache", action="store_true", help="Enable lesson-6 Paged KV cache")
    parser.add_argument("--no-sync", action="store_true", help="Skip cuda.synchronize timing (noisier, faster)")
    args = parser.parse_args()

    seed(0)

    llm = LLM(args.model)

    input_low = min(32, args.max_input_len)
    output_low = min(64, args.max_output_len)

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(input_low, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, max_tokens=randint(output_low, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    use_kv_cache = not args.no_kv_cache
    sync = not args.no_sync

    # warm up
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1), use_kv_cache=use_kv_cache, use_paged_kv_cache=args.paged_kv_cache)

    # ---- 正式 benchmark ----
    results = []
    for token_ids, sp in zip(prompt_token_ids, sampling_params):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=llm.device)
        results.append(run_one_sequence(llm, input_ids, sp, use_kv_cache, use_paged_kv_cache=args.paged_kv_cache, sync=sync))

    # ---- 汇总指标 ----
    total_output_tokens = sum(r["output_tokens"] for r in results)
    total_time = sum(r["total_time"] for r in results)
    total_prefill_time = sum(r["prefill_time"] for r in results)
    total_decode_time = sum(r["decode_time"] for r in results)

    # TTFT: 所有序列 prefill 的平均首 token 时延
    avg_ttft = total_prefill_time / len(results)

    # TPOT/ITL: 所有 decode step 的平均每 token 时延
    all_decode_times = [t for r in results for t in r["decode_times"]]
    avg_tpot = sum(all_decode_times) / len(all_decode_times) if all_decode_times else 0.0

    # 吞吐量
    throughput = total_output_tokens / total_time

    mode = "PAGED_KV_CACHE" if args.paged_kv_cache else ("NO_CACHE" if args.no_kv_cache else "KV_CACHE")
    print("=" * 64)
    print(f"[{mode}] AIOS Benchmark Results")
    print("=" * 64)
    print(f"  Sequences            : {len(results)}")
    print(f"  Total input tokens   : {sum(len(ids) for ids in prompt_token_ids)}")
    print(f"  Total output tokens  : {total_output_tokens}")
    print("-" * 64)
    print(f"  TTFT (mean)          : {avg_ttft * 1000:.2f} ms")
    print(f"  TPOT / ITL (mean)    : {avg_tpot * 1000:.2f} ms/token")
    print(f"  Prefill time (total) : {total_prefill_time:.2f} s")
    print(f"  Decode  time (total) : {total_decode_time:.2f} s")
    print(f"  Total time           : {total_time:.2f} s")
    print(f"  Throughput           : {throughput:.2f} tokens/s")
    print("=" * 64)


if __name__ == "__main__":
    main()
