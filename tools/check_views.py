"""Prueft Kamera-Konvention und Basis-PSNR gegen die LGM-Multiviews.

Der Feinschliff optimiert gegen die vier Ansichten, die LGM selbst erzeugt hat
(`workspace/mv/{name}_view*.png`). Dafuer muss die Kamera exakt stimmen -- sonst
optimiert man gegen ein falsch ausgerichtetes Ziel.

LGM nutzt `orbit_camera(elevation, azimuth, radius=1.5, opengl=True)`; gemessen
(align_scan.py) gehoert view i zum Azimut 90*i bei Radius 1.5.

Das Skript rendert mit beiden moeglichen Vertikal-Konventionen und meldet, welche
besser passt. Nur die bessere ist fuer den Feinschliff brauchbar.

    .venv\\Scripts\\python.exe tools\\check_views.py --name catstatue_rgba
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


def cam_from_orbit(azimuth, elevation=0.0, radius=1.5):
    """kiui-orbit -> (R, eye) in der Konvention von simready/depth.py."""
    pose = orbit_camera(elevation, azimuth, radius=radius, opengl=True)
    eye = torch.tensor(pose[:3, 3], dtype=torch.float32)
    right = torch.tensor(pose[:3, 0], dtype=torch.float32)
    up = torch.tensor(pose[:3, 1], dtype=torch.float32)
    back = torch.tensor(pose[:3, 2], dtype=torch.float32)
    R = torch.stack([right, -up, -back], dim=0)  # right, down, forward
    return R, eye


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-9 else 10 * np.log10(255.0**2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="catstatue_rgba")
    ap.add_argument("--workspace", default="LGM/workspace_test")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", default="out/views")
    args = ap.parse_args()

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    ply = os.path.join(args.workspace, args.name + ".ply")
    g = S.clean(S.load_gaussians(ply))
    print(f"[info] {len(g['xyz']):,} Gaussians")

    means = torch.tensor(g["xyz"], dtype=torch.float32, device=dev)
    scales = torch.tensor(g["scale"], dtype=torch.float32, device=dev)
    quats = torch.tensor(g["rot"], dtype=torch.float32, device=dev)
    opa = torch.tensor(g["opacity"], dtype=torch.float32, device=dev)
    col = torch.tensor(g["color"], dtype=torch.float32, device=dev)

    tiles = []
    for flip in (False, True):
        scores = []
        row = []
        for i in range(4):
            tgt = Image.open(os.path.join(args.workspace, "mv", f"{args.name}_view{i}.png"))
            tgt = tgt.convert("RGB").resize((args.size, args.size), Image.BILINEAR)
            tgt_np = np.asarray(tgt)

            R, eye = cam_from_orbit(90.0 * i)
            if flip:
                R = R.clone()
                R[1] = -R[1]

            img, alpha = DR.render(
                means, scales, quats, opa, col, R.to(dev), eye.to(dev), size=args.size
            )
            out = (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            scores.append(psnr(out, tgt_np))
            row.append(np.concatenate([out, tgt_np], axis=1))

        print(
            f"[{'flip' if flip else 'normal'}] PSNR je Ansicht: "
            + " ".join(f"{s:5.1f}" for s in scores)
            + f"  | Mittel {np.mean(scores):5.2f}"
        )
        tiles.append(np.concatenate(row, axis=1))

    os.makedirs(args.out, exist_ok=True)
    sheet = np.concatenate(tiles, axis=0)
    path = os.path.join(args.out, f"{args.name}_views.png")
    Image.fromarray(sheet).save(path)
    print(f"[out] {path}  (oben normal, unten gespiegelt; je Paar links Render, rechts Ziel)")


if __name__ == "__main__":
    main()
