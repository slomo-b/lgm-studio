"""Kann ImageDream andere Elevationen als 0?

Die Kamera-Konditionierung geht ueber `get_camera(num_frames, elevation=...)`
in mv_unet.py. Ob das *image*-konditionierte Modell damit umgehen kann, ist
eine andere Frage -- deshalb dieser Test: drei Durchlaeufe bei -25/0/+25 Grad,
gleicher Seed, und ein Vergleichsbild.

    .venv\\Scripts\\python.exe tools\\elev_probe.py --image LGM\\data_test\\catstatue_rgba.png
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "LGM"))

from core.local_backend import describe, get_device  # noqa: E402
from core.xpu_patches import apply as apply_xpu_patches  # noqa: E402

apply_xpu_patches()

from mvdream.mv_unet import MultiViewUNetModel  # noqa: E402
from mvdream.pipeline_mvdream import MVDreamPipeline  # noqa: E402

MODEL = "ashawkey/imagedream-ipmv-diffusers"


def prepare(path, size=256):
    import kiui
    from kiui.op import recenter

    img = kiui.read_image(path, mode="uint8")
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) * a + (1 - a) * 255.0
        img = np.concatenate([rgb, img[..., 3:4]], axis=-1).astype(np.uint8)
        mask = img[..., 3] > 0
    else:
        mask = np.ones(img.shape[:2], bool)
    img = recenter(img, mask, border_ratio=0.2)
    out = img[..., :3].astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        out = out * a + (1 - a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="LGM/data_test/catstatue_rgba.png")
    ap.add_argument("--elevs", default="-25,0,25")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/elev")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = get_device()
    print(f"[info] {describe(device)}")

    unet = MultiViewUNetModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16)
    pipe = MVDreamPipeline.from_pretrained(
        MODEL, unet=unet, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)

    image = prepare(args.image)
    rows = []
    for e in [float(x) for x in args.elevs.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        t0 = time.time()
        views = pipe(
            "",
            image,
            guidance_scale=5.0,
            num_inference_steps=args.steps,
            elevation=e,
            device=device,
        )
        if device.type == "xpu":
            torch.xpu.synchronize()
        mv = np.stack([views[1], views[2], views[3], views[0]], axis=0)
        rows.append(mv)
        print(f"[elev {e:+.0f}] {time.time() - t0:.1f}s")
        for i in range(4):
            import imageio.v2 as imageio

            imageio.imwrite(
                os.path.join(args.out, f"elev{e:+.0f}_view{i}.png"),
                (np.clip(mv[i], 0, 1) * 255).astype(np.uint8),
            )

    import imageio.v2 as imageio

    sheet = np.concatenate([np.concatenate([r[i] for i in range(4)], axis=1) for r in rows], axis=0)
    path = os.path.join(args.out, "grid.png")
    imageio.imwrite(path, (np.clip(sheet, 0, 1) * 255).astype(np.uint8))
    print(f"[out] {path}  (Zeilen: {args.elevs})")

    # Wie stark unterscheiden sich die Elevationen? Gleiche Ansicht, andere Kamera.
    if len(rows) > 1:
        base = rows[0]
        for k, r in enumerate(rows[1:], 1):
            d = float(np.abs(base - r).mean())
            print(f"[diff] elev0 vs elev{args.elevs.split(',')[k]}: mittlere Pixelabweichung {d:.3f}")


if __name__ == "__main__":
    main()
