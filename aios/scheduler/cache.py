from __future__ import annotations

import torch
from aios.kvcache import create_naive_cache_manager


class CacheManager:
    def __init__(self, device: torch.device, num_pages: int):
        self._free_slots = torch.arange(num_pages, dtype=torch.int32, device=device)
        self.device = device
        self.manager = create_naive_cache_manager(device=device)
        self.num_pages = num_pages

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            self._free_slots = torch.cat([self._free_slots, indices])

    def allocate(self, needed_len: int) -> torch.Tensor:
        if needed_len <= (free_len := len(self._free_slots)):
            allocated = self._free_slots[:needed_len]
            self._free_slots = self._free_slots[needed_len:]
            return allocated
        raise NotImplementedError("CacheManager eviction is not implemented.")

    