"""Prueft den differenzierbaren Renderer gegen den Referenz-Renderer.

Zwei Fragen: sieht das Bild richtig aus, und wie teuer sind Forward und
Backward auf der iGPU? Beides entscheidet, ob ein Feinschliff ohne CUDA
praktikabel ist.

    .venv\\Scripts\\python.exe tools\\test_diffrender.py --ply LGM\\workspace_test\\catstatue_rgba.ply
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simready import depth as D
from simready import diffrender as DR
from simready import splat as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", default="LGM/workspace_test/catstatue_rgba.ply")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="out/diffrender")
    args = ap.parse_args()

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    g = S.clean(S.load_gaussians(args.ply))
    print(f"[info] {len(g['xyz']):,} Gaussians, Device {dev}")

    means = torch.tensor(g["xyz"], dtype=torch.float32, device=dev, requires_grad=True)
    scales = torch.tensor(g["scale"], dtype=torch.float32, device=dev)
    quats = torch.tensor(g["rot"], dtype=torch.float32, device=dev)
    opa = torch.tensor(g["opacity"], dtype=torch.float32, device=dev)
    col = torch.tensor(g["color"], dtype=torch.float32, device=dev)

    # Blick von schraeg oben, wie im Vergleich
    a, e, r = np.deg2rad(30.0), np.deg2rad(12.0), 1.9
    eye = torch.tensor(
        [r * np.cos(e) * np.sin(a), r * np.sin(e), r * np.cos(e) * np.cos(a)],
        dtype=torch.float32,
        device=dev,
    )
    fwd = torch.nn.functional.normalize(-eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=dev), dim=0), dim=0)
    down = torch.cross(fwd, right, dim=0)
    R = torch.stack([right, down, fwd], dim=0)

    t0 = time.perf_counter()
    img, alpha = DR.render(means, scales, quats, opa, col, R, eye, size=args.size)
    if dev.type == "xpu":
        torch.xpu.synchronize()
    fwd_ms = (time.perf_counter() - t0) * 1000
    print(f"[ok] forward: {fwd_ms:.0f} ms | alpha-Mittel {alpha.mean().item():.3f}")

    t0 = time.perf_counter()
    (img.mean() - alpha.mean()).backward()
    if dev.type == "xpu":
        torch.xpu.synchronize()
    bwd_ms = (time.perf_counter() - t0) * 1000
    print(f"[ok] backward: {bwd_ms:.0f} ms | grad-Mittel {means.grad.abs().mean().item():.2e}")

    os.makedirs(args.out, exist_ok=True)
    ours = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    # Referenz: derselbe Blick mit dem nicht-differenzierbaren Renderer
    _, ref = D.render_depth(
        np.asarray(g["xyz"], np.float64),
        np.asarray(g["color"], np.float32),
        size=args.size,
        R=R.detach().cpu().numpy().astype(np.float64),
        eye=eye.detach().cpu().numpy().astype(np.float64),
    )

    from PIL import Image

    both = np.concatenate([ours, ref], axis=1)
    path = os.path.join(args.out, "vergleich.png")
    Image.fromarray(both).save(path)
    print(f"[out] {path}  (links: differenzierbar, rechts: Referenz)")

    per_step = fwd_ms + bwd_ms
    print(f"[info] ein Optimierungsschritt ~{per_step:.0f} ms -> 300 Schritte ~{per_step * 300 / 1000 / 60:.1f} min")


if __name__ == "__main__":
    main()
