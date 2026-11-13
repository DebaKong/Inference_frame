from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch

from ..core import Batch, SamplingParams
from ..engine.sample import Sampler


@dataclass
class _ReqState:
    """Scheduler-private per-request state."""

    req: "Req"  # forward ref to avoid circular import
    sampler: Sampler
    finished: bool = False


@dataclass
class ScheduledBatch:
    """Scheduler -> Engine execution unit."""

    batch: Batch
    samplers: List[Sampler]  # aligned with batch.reqs
    state_indices: List[int]  # row indices into running_reqs (decode batches)
    admitted_state: "_ReqState | None" = None  # set by PrefillManager for bsz=1 prefill


@dataclass
class _PendingReq:
    """A request waiting to be admitted to prefill."""

    input_ids: torch.Tensor
    sampling_params: SamplingParams
    uid: int

# Re-export Req for _ReqState.req resolution at type-checking time.
from ..core import Req  # noqa: E402