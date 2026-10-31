"""12 Views aus drei Elevations-Durchlaeufen, in splatcage-Format.

Die Kamera-Zuordnung ist nicht geraten: `get_camera(num_frames=4)` liefert
Azimute 0/90/180/270 in Modellreihenfolge, und infer.py rotiert die Liste um
eins (`[mv[1], mv[2], mv[3], mv[0]]`). Datei-Ansicht i entspricht deshalb
Azimut 90*(i+1).

    .venv\\Scripts\\python.exe tools\\gen_views_elev.py --image LGM\\data_test\\catstatue_rgba.png

Ausgabe:
    out/elev_views/views/<elev>_<i>.png    RGB, weisser Hintergrund
    out/elev_views/views.json              splatcage-Format (mask-mode luma)
"""

import argparse
import json
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
FOVY = 49.1


def prepare(path):
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


def w2c_opencv(azim_deg, elev_deg, radius):
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    eye = radius * np.array(
        [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], dtype=np.float64
    )
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = -R @ eye
    return w2c, eye


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="LGM/data_test/catstatue_rgba.png")
    ap.add_argument("--elevs", default="-25,0,25")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--out", default="out/elev_views")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "views"), exist_ok=True)
    device = get_device()
    print(f"[info] {describe(device)}")

    unet = MultiViewUNetModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16)
    pipe = MVDreamPipeline.from_pretrained(
        MODEL, unet=unet, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)
    image = prepare(args.image)

    import imageio.v2 as imageio

    entries = []
    t_all = time.time()
    for e in [float(x) for x in args.elevs.split(",")]:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        t0 = time.time()
        views = pipe(
            "", image, guidance_scale=5.0,
            num_inference_steps=args.steps, elevation=e, device=device,
        )
        if device.type == "xpu":
            torch.xpu.synchronize()
        mv = np.stack([views[1], views[2], views[3], views[0]], axis=0)
        print(f"[elev {e:+.0f}] 4 Views in {time.time() - t0:.1f}s")

        for i in range(4):
            azim = (90.0 * (i + 1)) % 360.0
            name = f"e{e:+.0f}_v{i}.png"
            imageio.imwrite(
                os.path.join(args.out, "views", name),
                (np.clip(mv[i], 0, 1) * 255).astype(np.uint8),
            )
            w2c, _ = w2c_opencv(azim, e, args.radius)
            h, w = mv[i].shape[:2]
            f = (h * 0.5) / np.tan(0.5 * np.deg2rad(FOVY))
            entries.append({
                "mask": f"views/{name}",
                "width": w,
                "height": h,
                "fx": float(f),
                "fy": float(f),
                "cx": w * 0.5,
                "cy": h * 0.5,
                "world_to_cam": w2c.tolist(),
                "azim_deg": azim,
                "elev_deg": e,
            })

    meta = {
        "generator": "imagedream-3elev",
        "source": os.path.abspath(args.image),
        "steps": args.steps,
        "radius": args.radius,
        "views": entries,
    }
    path = os.path.join(args.out, "views.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[out] {path}: {len(entries)} Views in {time.time() - t_all:.0f}s gesamt")


if __name__ == "__main__":
    main()
