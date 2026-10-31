"""TripoSplat lokal auf der Intel-iGPU.

Das Modell ist fuer CUDA gedacht (`device="cuda"` als Default), nimmt den
Device aber als Parameter. Zwei Risiken auf dieser Plattform:

  1. `model.py` nutzt index_select / scatter_add_ / Boolean-Indexing -- genau
     die Index-Kernel, die hier laut SETUP.md haengen. `core.xpu_patches`
     umgeht einige davon, deshalb wird es vorher angewandt.
  2. bfloat16 muss die iGPU koennen (Arc kann es).

    .venv\\Scripts\\python.exe tools\\triposplat_local.py --device xpu --gaussians 32768
    .venv\\Scripts\\python.exe tools\\triposplat_local.py --device cpu --gaussians 16384 --steps 8
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--image", default=os.path.join(ROOT, "LGM", "data_test", "catstatue_rgba.png"))
ap.add_argument("--device", default="xpu")
ap.add_argument("--gaussians", type=int, default=32768)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--out", default=os.path.join(ROOT, "out", "triposplat"))
ap.add_argument("--no-patches", action="store_true")
args = ap.parse_args()

sys.path.insert(0, os.path.join(ROOT, "LGM"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "TripoSplat"))

import torch  # noqa: E402

if not args.no_patches:
    try:
        from core.xpu_patches import apply as apply_xpu_patches

        apply_xpu_patches()
        print("[info] xpu_patches angewandt")
    except Exception as e:
        print(f"[warn] xpu_patches nicht angewandt: {e}")

print(f"[info] torch {torch.__version__} | xpu verfügbar: {hasattr(torch, 'xpu') and torch.xpu.is_available()}")
if args.device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
    print("[warn] kein XPU -> CPU")

from triposplat import TripoSplatPipeline  # noqa: E402

CK = os.path.join(ROOT, "vendor", "TripoSplat", "ckpts")
paths = {
    "ckpt_path": os.path.join(CK, "diffusion_models", "triposplat_fp16.safetensors"),
    "decoder_path": os.path.join(CK, "vae", "triposplat_vae_decoder_fp16.safetensors"),
    "dinov3_path": os.path.join(CK, "clip_vision", "dino_v3_vit_h.safetensors"),
    "flux2_vae_encoder_path": os.path.join(CK, "vae", "flux2-vae.safetensors"),
    "rmbg_path": os.path.join(CK, "background_removal", "birefnet.safetensors"),
}
missing = [k for k, v in paths.items() if not os.path.exists(v)]
if missing:
    raise SystemExit(f"Gewichte fehlen: {missing}")

t0 = time.perf_counter()
pipe = TripoSplatPipeline(**paths, device=args.device)
print(f"[load] {time.perf_counter() - t0:.1f}s")

os.makedirs(args.out, exist_ok=True)
t0 = time.perf_counter()
gaussian, prepared = pipe.run(
    args.image,
    seed=args.seed,
    steps=args.steps,
    num_gaussians=args.gaussians,
    show_progress=True,
)
if args.device == "xpu":
    torch.xpu.synchronize()
dt = time.perf_counter() - t0
print(f"[run] {dt:.1f}s für {args.gaussians} Gaussians, {args.steps} Schritte")

stem = os.path.splitext(os.path.basename(args.image))[0]
ply = os.path.join(args.out, f"{stem}_{args.gaussians}.ply")
gaussian.save_ply(ply)
print(f"[out] {ply}")
print(f"[info] Speicher: {torch.xpu.memory_allocated()/1e9:.2f} GB belegt" if args.device == "xpu" else "")
