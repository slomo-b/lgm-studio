"""Sucht die Kamera, mit der das PLY die LGM-Multiviews wirklich trifft.

Die naive Annahme (Azimut 90/180/270/0, Radius 1.5) liefert nur ~11.7 dB. Also
systematisch abtasten: Azimut in 30-Grad-Schritten, mehrere Radien -- und
zusaetzlich pruefen, ob das Ziel horizontal gespiegelt ist.

    .venv\\Scripts\\python.exe tools\\align_scan.py --name catstatue_rgba
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiui.cam import orbit_camera  # noqa: E402

from simready import diffrender as DR  # noqa: E402
from simready import splat as S  # noqa: E402


def cam(azimuth, radius, elevation=0.0):
    pose = orbit_camera(elevation, azimuth, radius=radius, opengl=True)
    eye = torch.tensor(pose[:3, 3], dtype=torch.float32)
    right = torch.tensor(pose[:3, 0], dtype=torch.float32)
    up = torch.tensor(pose[:3, 1], dtype=torch.float32)
    back = torch.tensor(pose[:3, 2], dtype=torch.float32)
    return torch.stack([right, -up, -back], dim=0), eye


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-9 else 10 * np.log10(255.0**2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="catstatue_rgba")
    ap.add_argument("--workspace", default="LGM/workspace_test")
    ap.add_argument("--view", type=int, default=0)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    g = S.clean(S.load_gaussians(os.path.join(args.workspace, args.name + ".ply")))
    means = torch.tensor(g["xyz"], dtype=torch.float32, device=dev)
    scales = torch.tensor(g["scale"], dtype=torch.float32, device=dev)
    quats = torch.tensor(g["rot"], dtype=torch.float32, device=dev)
    opa = torch.tensor(g["opacity"], dtype=torch.float32, device=dev)
    col = torch.tensor(g["color"], dtype=torch.float32, device=dev)

    tgt = Image.open(
        os.path.join(args.workspace, "mv", f"{args.name}_view{args.view}.png")
    ).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
    tgt = np.asarray(tgt)

    print(f"[info] {len(g['xyz']):,} Gaussians, Ziel view{args.view}", flush=True)
    rows = []
    for radius in (1.2, 1.5, 1.8, 2.1, 2.5):
        scores = []
        for az in range(0, 360, 30):
            R, eye = cam(az, radius)
            img, _ = DR.render(means, scales, quats, opa, col, R.to(dev), eye.to(dev), size=args.size)
            out = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            s = psnr(out, tgt)
            s_mirror = psnr(out[:, ::-1], tgt)
            scores.append((az, s, s_mirror))
        best = max(scores, key=lambda t: max(t[1], t[2]))
        line = " ".join(f"{s:5.1f}" for _, s, _ in scores)
        print(f"[r={radius}] {line}   | best az={best[0]} {max(best[1], best[2]):.1f} dB"
              f"{' (gespiegelt)' if best[2] > best[1] else ''}", flush=True)
        rows.append((radius, scores))

    r, scores = max(rows, key=lambda t: max(max(s, sm) for _, s, sm in t[1]))
    az, s, sm = max(scores, key=lambda t: max(t[1], t[2]))
    print(f"\n[beste] radius={r} azimut={az} -> {max(s, sm):.2f} dB", flush=True)


if __name__ == "__main__":
    main()
