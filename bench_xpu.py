"""Measure whether this machine can run LGM's diffusion stack.

Run with the XPU-enabled interpreter:
    .venv\\Scripts\\python.exe bench_xpu.py

Prints device info and a throughput estimate for the two hot spots:
  1. the MVDream/ImageDream UNet (spatial self-attention at 32x32, batch 4)
  2. LGM's own cross-attention blocks
"""

import time

import torch
import torch.nn.functional as F


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def bench(fn, device, warmup=2, iters=5, label=""):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    print(f"    {label:<34s} {dt * 1000:8.1f} ms")
    return dt


def main():
    dev = pick_device()
    print(f"device      : {dev}")
    if dev.type == "xpu":
        print(f"name        : {torch.xpu.get_device_name(0)}")
        print(f"properties  : {torch.xpu.get_device_properties(0)}")
        print(f"mem_alloc   : {torch.xpu.mem_get_info()}")
    print(f"torch       : {torch.__version__}")
    print(f"threads     : {torch.get_num_threads()} (cpu)")
    print()

    dt = torch.float16 if dev.type != "cpu" else torch.float32

    # --- 1. spatial self-attention, as in the MVDream UNet at 256px -------
    # batch 4 views, 1024 tokens (=32x32 latent), 8 heads, dim 64
    print("self-attention 4x1024 tok x 8 heads:")
    q = torch.randn(4 * 8, 1024, 64, device=dev, dtype=dt)
    bench(lambda: F.scaled_dot_product_attention(q, q, q), dev, label="SDPA")

    # --- 2. a 256->256 residual conv block at 64x64 latent ---------------
    print("conv 320->320 @ 64x64 x batch 4:")
    conv = torch.nn.Conv2d(320, 320, 3, padding=1).to(dev, dt)
    x = torch.randn(4, 320, 64, 64, device=dev, dtype=dt)
    bench(lambda: conv(x), dev, label="conv3x3")

    # --- 3. grouped norm + silu (elementwise, bandwidth bound) -----------
    print("elementwise 4x320x64x64:")
    bench(lambda: F.silu(x).mean(), dev, label="silu+mean")

    print()
    print("Rough projection for LGM 'big' (ImageDream 30 steps + LGM decoder):")
    print("  measured above, extrapolate manually -- see SETUP.md")


if __name__ == "__main__":
    main()
