"""Vergleicht Sampler und Schrittweiten der Multiview-Diffusion.

Fragestellung: wie wenige Schritte reichen, ohne dass die vier Ansichten
schlechter werden? Gemessen gegen den 30-Schritte-DDIM-Lauf mit identischem
Seed, auf denselben Eingabebildern.

    .venv\\Scripts\\python.exe bench_steps.py --image LGM\\data_test\\catstatue_rgba.png

Ausgabe: out/steps/vergleich.png und eine Tabelle mit PSNR / SSIM / LPIPS / Zeit.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LGM"))

from core.local_backend import get_device, get_autocast_dtype, model_dtype, describe
from core.xpu_patches import apply as apply_xpu_patches

apply_xpu_patches()

import kiui
from kiui.op import recenter
from mvdream.mv_unet import MultiViewUNetModel
from mvdream.pipeline_mvdream import MVDreamPipeline

MODEL = "ashawkey/imagedream-ipmv-diffusers"


def load_pipe(device):
    unet = MultiViewUNetModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16)
    pipe = MVDreamPipeline.from_pretrained(
        MODEL, unet=unet, torch_dtype=torch.float16, trust_remote_code=True
    )
    return pipe.to(device)


def prepare_image(path):
    """RGBA lesen, auf weiss komponieren, recentern -- wie infer.py, nur ohne rembg."""
    img = kiui.read_image(path, mode="uint8")
    if img.shape[-1] == 4:
        alpha = img[..., 3] > 0
        flat = img[..., :3].astype(np.float32) * (img[..., 3:4] / 255.0)
        flat = flat + (1.0 - img[..., 3:4] / 255.0) * 255.0
        img = np.concatenate([flat, img[..., 3:4]], axis=-1).astype(np.uint8)
        mask = alpha
    else:
        mask = np.ones(img.shape[:2], dtype=bool)
    img = recenter(img, mask, border_ratio=0.2)
    out = img[..., :3].astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        out = out * a + (1.0 - a)
    return out


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(1.0 / mse))


def ssim(a, b):
    from skimage.metrics import structural_similarity

    vals = [
        structural_similarity(a[i], b[i], channel_axis=-1, data_range=1.0)
        for i in range(a.shape[0])
    ]
    return float(np.mean(vals))


def make_lpips(device):
    try:
        from kiui.lpips import LPIPS

        net = LPIPS(net="vgg").to(device)
        net.eval()

        def fn(a, b):
            ta = torch.from_numpy(a).permute(0, 3, 1, 2).to(device) * 2 - 1
            tb = torch.from_numpy(b).permute(0, 3, 1, 2).to(device) * 2 - 1
            with torch.no_grad():
                return float(net(ta, tb).mean())

        return fn
    except Exception as e:
        print(f"[warn] LPIPS nicht verfuegbar: {e}")
        return lambda a, b: float("nan")


def run(pipe, image, device, steps, solver="ddim", seed=0):
    from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler

    base_cfg = dict(pipe.scheduler.config)
    if solver == "dpm":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            base_cfg, algorithm_type="dpmsolver++", solver_order=2
        )
    else:
        pipe.scheduler = DDIMScheduler.from_config(base_cfg)

    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    t0 = time.time()
    views = pipe(
        "",
        image,
        guidance_scale=5.0,
        num_inference_steps=int(steps),
        elevation=0,
        generator=gen,
        device=device,
    )
    if hasattr(torch, "xpu") and device.type == "xpu":
        torch.xpu.synchronize()
    dt = time.time() - t0
    views = np.stack([views[1], views[2], views[3], views[0]], axis=0)  # wie infer.py
    return views.astype(np.float32), dt


def grid(rows, labels, path):
    import imageio

    tiles = []
    for views in rows:
        strip = np.concatenate([views[i] for i in range(views.shape[0])], axis=1)
        tiles.append(strip)
    sheet = np.concatenate(tiles, axis=0)
    imageio.imwrite(path, (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
    print(f"[out] {path}  (Zeilen: {', '.join(labels)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="LGM/data_test/catstatue_rgba.png")
    ap.add_argument("--steps", default="30,24,20,16,12,8")
    ap.add_argument("--solvers", default="ddim")
    ap.add_argument("--baseline", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/steps")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"[info] {describe(device)}")

    image = prepare_image(args.image)
    pipe = load_pipe(device)
    lpips = make_lpips(device)

    base_views, base_dt = run(pipe, image, device, args.baseline, "ddim", args.seed)
    import imageio.v2 as imageio

    imageio.imwrite(os.path.join(args.out, f"base_{args.baseline}.png"),
                 (np.concatenate([base_views[i] for i in range(4)], axis=1) * 255).astype(np.uint8))
    print(f"[base] {args.baseline} Steps DDIM: {base_dt:.1f} s")

    rows, labels = [base_views], [f"{args.baseline} ddim"]
    print(f"\n{'variante':>14} {'zeit s':>8} {'speedup':>8} {'PSNR':>7} {'SSIM':>7} {'LPIPS':>7}")
    print("-" * 62)
    for solver in args.solvers.split(","):
        for s in [int(x) for x in args.steps.split(",")]:
            if solver == "ddim" and s == args.baseline:
                continue
            try:
                views, dt = run(pipe, image, device, s, solver, args.seed)
            except Exception as e:
                print(f"{solver + ' ' + str(s):>14}  FEHLER: {e}")
                continue
            rows.append(views)
            labels.append(f"{s} {solver}")
            print(
                f"{solver + ' ' + str(s):>14} {dt:>8.1f} {base_dt / dt:>7.1f}x"
                f" {psnr(base_views, views):>7.2f} {ssim(base_views, views):>7.3f}"
                f" {lpips(base_views, views):>7.3f}"
            )
            imageio.imwrite(
                os.path.join(args.out, f"{solver}_{s}.png"),
                (np.concatenate([views[i] for i in range(4)], axis=1) * 255).astype(np.uint8),
            )

    grid(rows, labels, os.path.join(args.out, "vergleich.png"))


if __name__ == "__main__":
    main()

