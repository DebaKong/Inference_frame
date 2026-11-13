from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch

from .common import ScheduledBatch, _ReqState, _PendingReq
from ..core import Batch, Req, SamplingParams
from ..engine.sample import Sampler

if TYPE_CHECKING:
    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager


@dataclass
class PrefillManager:
    """
    Holds the pending queue; admits one request per step as a bsz=1 prefill batch.

    Aligned with mini-sglang's PrefillManager, with two deliberate simplifications:
      1. bsz=1 admission (current attention kernel does not support varlen prefill).
      2. No chunked prefill (long-prompt chunking is out of scope).
    """

    cache_manager: "CacheManager"
    table_manager: "TableManager"
    decode_manager: "DecodeManager"
    device: torch.device
    pending_list: List[_PendingReq] = field(default_factory=list)

    def add_one_req(self, pending:_PendingReq) -> None:
        self.pending_list.append(pending)

    @property
    def runnable(self) -> bool:
        return bool(self.pending_list)

    def _can_admit(self, pending: _PendingReq, max_running: int) -> bool:
        # Capacity: one free table slot + one free running-set slot.
        if self.table_manager.available_size == 0:
            return False
        if len(self.decode_manager.running_reqs) >= max_running:
            return False
        prompt_len = len(pending.input_ids)
        needed = prompt_len + pending.sampling_params.max_tokens
        reserved = self.decode_manager.inflight_tokens
        free_pages = len(self.cache_manager._free_slots)
        return (needed + reserved) <= free_pages

    def schedule_next_batch(self, max_running: int) -> ScheduledBatch | None:
        if not self.pending_list:
            return None

        head = self.pending_list[0]
        if not self._can_admit(head, max_running):
            return None
        self.pending_list.pop(0)

        # Allocate table slot + prompt pages; write prompt into token_pool.
        table_idx = self.table_manager.allocate()
        prompt_len = len(head.input_ids)
        pages = self.cache_manager.allocate(prompt_len)
        self.table_manager.page_table[table_idx, : prompt_len] = pages
        self.table_manager.token_pool[table_idx, : prompt_len] = head.input_ids.to(torch.int32)

        req = Req(
            input_ids=head.input_ids,
            cached_len=0,
            output_len=head.sampling_params.max_tokens,
            uid=head.uid,
            sampling_params=head.sampling_params,
            table_idx=table_idx,
        )
        state = _ReqState(req=req, sampler=Sampler(head.sampling_params))

        # Build bsz=1 prefill batch
        input_ids = head.input_ids.to(self.device).long().unsqueeze(0) # (1, prompt_len)
        positions = torch.arange(prompt_len, device=self.device).unsqueeze(0) # (1, prompt_len)
        out_loc = self.table_manager.page_table[table_idx, : prompt_len].unsqueeze(0)

        batch = Batch(
            reqs=[req],
            phase="prefill",
            input_ids=input_ids,
            positions=positions,
            out_loc=out_loc,
            page_table=self.table_manager.page_table,
        )

        return ScheduledBatch(
            batch=batch,
            samplers=[state.sampler],
            state_indices=[0],
            admitted_state=state,
        )
    