"""Shared automatic inference-device policy for the portable product."""
from __future__ import annotations

import os
from typing import Any


def select_torch_device(
    *,
    minimum_free_gib: float = 1.0,
    force_cpu: bool = False,
    preference: str = "auto",
) -> tuple[Any, dict[str, Any]]:
    """Use CUDA only when it is available and has enough free memory.

    ``DUBCLEAN_DEVICE=cpu|cuda|auto`` is an expert override.  Automatic mode
    is the portable default and falls back to CPU for CPU-only Torch builds,
    unsupported adapters and a currently occupied GPU.
    """
    import torch

    override = str(
        os.environ.get("DUBCLEAN_DEVICE", preference or "auto")
    ).strip().casefold()
    if override not in {"auto", "cpu", "cuda"}:
        override = "auto"
    if force_cpu or override == "cpu":
        return torch.device("cpu"), {
            "device": "cpu",
            "reason": "forced_cpu",
            "minimum_free_gib": float(minimum_free_gib),
        }
    if not torch.cuda.is_available():
        return torch.device("cpu"), {
            "device": "cpu",
            "reason": "cuda_unavailable",
            "minimum_free_gib": float(minimum_free_gib),
        }
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (RuntimeError, OSError):
        free_bytes, total_bytes = 0, 0
    free_gib = float(free_bytes) / (1024.0**3)
    total_gib = float(total_bytes) / (1024.0**3)
    enough_memory = free_gib >= float(minimum_free_gib)
    if override != "cuda" and not enough_memory:
        return torch.device("cpu"), {
            "device": "cpu",
            "reason": "cuda_memory_busy",
            "free_gib": free_gib,
            "total_gib": total_gib,
            "minimum_free_gib": float(minimum_free_gib),
        }
    return torch.device("cuda"), {
        "device": "cuda",
        "reason": "forced_cuda" if override == "cuda" else "auto_cuda",
        "name": torch.cuda.get_device_name(0),
        "free_gib": free_gib,
        "total_gib": total_gib,
        "minimum_free_gib": float(minimum_free_gib),
    }


def device_status_text(details: dict[str, Any]) -> str:
    if details.get("device") == "cuda":
        return f"GPU: {details.get('name') or 'CUDA'}"
    if details.get("reason") == "cuda_memory_busy":
        return "CPU: видеокарта занята"
    return "CPU"
