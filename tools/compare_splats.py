"""Vergleicht mehrere Splat-Dateien aus identischen Kameras.

Gleiche Posen, gleiche Bildweite, gleiche Reihenfolge -- sonst vergleicht man
Kamerawinkel statt Modelle. Beide Formate (LGM und TripoSplat) sind 3DGS-PLYs;
`simready.splat` liest nur x/y/z, f_dc, opacity, scale, rot und ignoriert
hoehere SH-Baender, was fuer eine Sichtpruefung reicht.

    .venv-simready\\Scripts\\python.exe tools\\compare_splats.py ^
        --ply lgm=LGM\\workspace_test\\catstatue_rgba.ply ^
        --ply tripo=out\\triposplat\\catstatue_rgba_32768.ply ^
        --out out\\cmp
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simready import depth as D
from simready import splat as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", action="append", required=True, help="label=pfad.ply")
    ap.add_argument("--out", default="out/cmp")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--views", type=int, default=3)
    ap.add_argument("--elev", type=float, default=12.0)
    ap.add_argument("--radius", type=float, default=1.9)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    items = []
    for spec in args.ply:
        label, path = spec.split("=", 1)
        if not os.path.exists(path):
            print(f"[warn] {label}: {path} fehlt, uebersprungen")
            continue
        g = S.clean(S.load_gaussians(path))
        xyz = np.asarray(g["xyz"], dtype=np.float64)
        col = np.asarray(g["color"], dtype=np.float32)
        ext = xyz.max(0) - xyz.min(0)
        print(
            f"[{label}] {len(xyz):,} Gaussians | Ausdehnung "
            f"{ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}"
        )
        items.append((label, xyz, col))

    if not items:
        raise SystemExit("keine lesbaren PLYs")

    # gemeinsames Zentrum und gemeinsamer Radius, damit die Bilder vergleichbar sind.
    # Box-Mitte statt Mittelwert: ein Mittelwert verschiebt sich, wenn ein Modell
    # mehr Punkte an einer Seite hat. Radius aus der Box-Diagonale, damit nichts
    # abgeschnitten wird (sichtbare Halbhoehe bei 49.1 Grad ist 0.457 * radius).
    all_pts = np.concatenate([it[1] for it in items])
    lo = all_pts.min(axis=0)
    hi = all_pts.max(axis=0)
    center = (lo + hi) / 2.0
    diag = float(np.linalg.norm(hi - lo))
    radius = max(diag / 0.914 * 1.08, 1e-3)

    azims = [30 + 360.0 * i / args.views for i in range(args.views)]
    tiles = []
    for label, xyz, col in items:
        row = []
        for az in azims:
            a, e = np.deg2rad(az), np.deg2rad(args.elev)
            eye = center + radius * np.array(
                [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)]
            )
            fwd = (center - eye) / np.linalg.norm(center - eye)
            right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
            right /= np.linalg.norm(right)
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd], axis=0)
            _, rgb = D.render_depth(xyz, col, size=args.size, R=R, eye=eye)
            # leere Pixel auf Weiss setzen -- sonst schwarzer Hintergrund, und
            # ein helles Objekt auf Schwarz wirkt anders als auf Weiss
            empty = rgb.sum(axis=2) == 0
            rgb[empty] = 255
            row.append(rgb)
        tiles.append((label, np.concatenate(row, axis=1)))

    from PIL import Image, ImageDraw

    h, w = tiles[0][1].shape[:2]
    label_h = 22
    sheet = Image.new("RGB", (w, (h + label_h) * len(tiles)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(tiles):
        y = i * (h + label_h)
        sheet.paste(Image.fromarray(img), (0, y + label_h))
        draw.text((6, y + 5), f"{label}   (az {', '.join(f'{a:.0f}' for a in azims)})", fill="black")

    path = os.path.join(args.out, "vergleich.png")
    sheet.save(path)
    print(f"[out] {path}")


if __name__ == "__main__":
    main()
