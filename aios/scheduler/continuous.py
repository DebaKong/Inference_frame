from __future__ import annotations

from typing import Any, List

import torch

from ..core import SamplingParams
from .cache import CacheManager
from .common import ScheduledBatch, _PendingReq, _ReqState
from .decode import DecodeManager
from .prefill import PrefillManager
from .table import TableManager


class ContinuousScheduler:
    """
    Continuous-batching scheduler.

    Composes PrefillManager + DecodeManager, aligned with mini-sglang's top-level Scheduler:
      - prefill_manager.schedule_next_batch(max_running) is tried first (admit one).
      - decode_manager.schedule_next_batch() runs otherwise over the full running set.
      - On completion, resources are freed immediately via _free_req_resources so that a
        new pending request can reuse the slot / pages in the next iteration.

    Deliberate simplifications vs mini-sglang:
      1. bsz=1 prefill (attention kernel does not support varlen prefill yet).
      2. No chunked prefill (long-prompt splitting deferred).
      3. No prefix caching (direct page free).
      4. Single CUDA stream, no overlap loop.
    """

    def __init__(
            self,
            table_manager: TableManager,
            cache_manager: CacheManager,
            eos_token_id: int,
            device: torch.device,
            max_running_reqs: int,
    ) -> None:
        self.table_manager = table_manager
        self.cache_manager = cache_manager
        self.eos_token_id = eos_token_id
        self.device = device
        self.max_running = max_running_reqs

        self.decode_manager = DecodeManager(
            cache_manager=cache_manager,
            table_manager=table_manager,
            device=device,
        )
        self.prefill_manager = PrefillManager(
            cache_manager=cache_manager,
            table_manager=table_manager,
            decode_manager=self.decode_manager,
            device=device,
        )
        self.finished: List[_ReqState] = []
        self._next_uid = 0

    # admission
    def add_request(
            self,
            input_ids: torch.Tensor,
            sampling_params: SamplingParams,
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self.prefill_manager.add_one_req(
            _PendingReq(input_ids=input_ids, sampling_params=sampling_params, uid=uid)
        )
        return uid

    # scheduling
    def schedule_next_batch(self) -> ScheduledBatch | None:
        # Prefill-first policy (matches mini-sglang default).
        return (
            self.prefill_manager.schedule_next_batch(self.max_running)
            or self.decode_manager.schedule_next_batch()
        )

    # post-processing
    def process_batch_output(
            self,
            scheduled: ScheduledBatch,
            next_tokens: torch.Tensor,
    ) -> None:
        tokens = next_tokens.view(-1).tolist()

        if scheduled.batch.is_prefill:
            state = scheduled.admitted_state
            assert state is not None, "prefill ScheduledBatch must set admitted_state"
            self._advance(state, tokens[0])
            if state.finished:
                self._free_req_resources(state)
                self.finished.append(state)
            else:
                self.decode_manager.add_req(state)
        else:
            for state, tok in zip(self.decode_manager.running_reqs, tokens):
                self._advance(state, tok)
            for state in self.decode_manager.filter_reqs():
                self._free_req_resources(state)
                self.finished.append(state)

    def _advance(self, state: _ReqState, tok: int) -> None:
        req = state.req
        req.complete_one()
        self.table_manager.token_pool[req.table_idx, req.device_len - 1] = tok
        req.generated.append(tok)

        hit_eos = (not req.sampling_params.ignore_eos) and (tok == self.eos_token_id)
        if hit_eos or not req.can_decode():
            state.finished = True

    def _free_req_resources(self, state: _ReqState) -> None:
        req = state.req
        used_pages = self.table_manager.page_table[req.table_idx, : req.cached_len]
        self.cache_manager._free(used_pages)
        self.table_manager.free(req.table_idx)

    # inspection
    @property
    def has_work(self) -> bool:
        return self.prefill_manager.runnable or self.decode_manager.runnable

    def collect_results(self, tokenizer) -> list[dict]:
        results = [
            {
                "uid": s.req.uid,
                "token_ids": s.req.generated,
                "text": tokenizer.decode(s.req.generated, skip_special_tokens=True),
            }
            for s in self.finished
        ]
        results.sort(key=lambda r: r["uid"])
        return results
    