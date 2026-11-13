from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import torch

from ..core import Batch
from .common import ScheduledBatch, _ReqState

if TYPE_CHECKING:
    from .cache import CacheManager
    from .table import TableManager


@dataclass
class DecodeManager:
    """
    Holds the running set; builds one decode batch per step over all live reqs.

    Uses a List (not Set) to preserve admission order for deterministic batching
    and easier debugging.
    """

    cache_manager: "CacheManager"
    table_manager: "TableManager"
    device: torch.device
    running_reqs: List[_ReqState] = field(default_factory=list)

    def add_req(self, state: _ReqState) -> None:
        self.running_reqs.append(state)

    def filter_reqs(self) -> List[_ReqState]:
        """Remove finished reqs from the running set, return them."""
        finished = [s for s in self.running_reqs if s.finished]
        if finished:
            self.running_reqs = [s for s in self.running_reqs if not s.finished]
        return finished

    @property
    def inflight_tokens(self) -> int:
        """Conservative upper bound on tokens the running set will still generate."""
        return sum(s.req.remain_len for s in self.running_reqs)

    @property
    def runnable(self) -> bool:
        return bool(self.running_reqs)
    
    def schedule_next_batch(self) -> ScheduledBatch | None:
        if not self.running_reqs:
            return None

        reqs = [s.req for s in self.running_reqs]
        samplers = [s.sampler for s in self.running_reqs]
        B = len(reqs)

        table_idxs = torch.tensor(
            [r.table_idx for r in reqs], device=self.device, dtype=torch.long
        )
        position_id = torch.tensor(
            [r.cached_len for r in reqs], device=self.device, dtype=torch.long
        )

        # Allocate one page per req (this step's KV write slot, at position cached_len).
        new_pages = self.cache_manager.allocate(B)
        self.table_manager.page_table[table_idxs, position_id] = new_pages

        # Input token = token_pool[cached_len] (written by previous step or by prefill).
        input_ids = self.table_manager.token_pool[table_idxs, position_id].long().unsqueeze(1)
        positions = position_id.unsqueeze(1) # (B, 1)
        out_loc = new_pages.unsqueeze(1) # (B, 1)

        batch = Batch(
            reqs=reqs,
            phase="decode",
            input_ids=input_ids,
            positions=positions,
            out_loc=out_loc,
            page_table=self.table_manager.page_table,
        )

        return ScheduledBatch(
            batch=batch,
            samplers=samplers,
            state_indices=list(range(B)),
            admitted_state=None,
        )
    