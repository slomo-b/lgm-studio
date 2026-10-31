"""[LOCAL PATCH] Backend detection so LGM can run without an NVIDIA GPU.

Upstream LGM assumes CUDA everywhere. These helpers pick the best available
device (CUDA -> Intel XPU -> Apple MPS -> CPU) and the matching autocast dtype.
"""

import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    # fp16 is native on Intel Arc iGPUs and NVIDIA cards; CPU only really
    # benefits from bf16.
    if device.type == "cpu":
        return torch.bfloat16
    return torch.float16


def model_dtype(device: torch.device) -> torch.dtype:
    # The fp16 LGM checkpoint is kept half precision on any accelerator,
    # but promoted back to fp32 on CPU where fp16 matmul is emulated/slow.
    return torch.float32 if device.type == "cpu" else torch.float16


def describe(device: torch.device) -> str:
    if device.type == "xpu":
        idx = device.index or 0
        name = torch.xpu.get_device_name(idx)
        props = torch.xpu.get_device_properties(idx)
        total = getattr(props, "total_memory", None)
        mem = f", {total / 2**30:.1f} GB" if total else ""
        return f"Intel XPU [{idx}] {name}{mem}"
    if device.type == "cuda":
        idx = device.index or 0
        name = torch.cuda.get_device_name(idx)
        total = torch.cuda.get_device_properties(idx).total_memory
        return f"CUDA [{idx}] {name}, {total / 2**30:.1f} GB"
    if device.type == "mps":
        return "Apple MPS"
    return "CPU"
