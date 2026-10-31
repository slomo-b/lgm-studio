"""Trockenlauf: aus einem vorhandenen Splat einen Orbit-Datensatz rendern.

Damit testen wir die Naht, an der spaeter SEVA steht, ohne SEVA zu brauchen.
Der Renderer ist derselbe, der auch den TSDF speist -- was hier herauskommt,
hat exakt die Form, die ein Multi-View-Generator liefert: N Bilder plus Posen.

    .venv-simready\\Scripts\\python.exe tools\\make_orbit_views.py ^
        --ply LGM\\workspace_test\\catstatue_rgba.ply --views 12 --out out\\orbit_test

Ausgabe:
    out\\orbit_test\\views\\000.png ...   RGBA, Alpha = Objektmaske
    out\\orbit_test\\views.json           splatcage-Format (world_to_cam, OpenCV)
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from simready import depth as D
from simready import splat as S


def orbit_cams(n, elev_lo, elev_hi, radius, size, fovy):
    """Gleichmaessig ueber Azimut, gestreut ueber Elevation.

    Der Elevations-Spread ist der eigentliche Punkt: bei nur einer Hoehe kann
    keine Silhouette die Hochachse einschraenken.
    """
    cams = []
    for i in range(n):
        azim = 360.0 * i / n
        # zwei Durchlaeufe: untere und obere Haelfte, damit oben/unten
        # gleich viele Ansichten bekommen
        t = (i * 0.61803398875) % 1.0
        elev = elev_lo + (elev_hi - elev_lo) * t
        R, eye = D.orbit_cameras(1, elev_range=(elev, elev), radius=radius)[0]
        # orbit_cameras nutzt den goldenen Winkel fuer den Azimut -- hier wollen
        # wir ihn vorgeben, deshalb rotieren wir die Kamera um Y nach.
        a = np.deg2rad(azim)
        e = np.deg2rad(elev)
        eye = radius * np.array([np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)])
        fwd = -eye / np.linalg.norm(eye)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], axis=0)
        cams.append((R, eye, azim, elev))
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--radius", type=float, default=1.9)
    ap.add_argument("--fovy", type=float, default=D.FOVY)
    ap.add_argument("--elev-lo", type=float, default=-25.0)
    ap.add_argument("--elev-hi", type=float, default=35.0)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "views"), exist_ok=True)
    g = S.clean(S.load_gaussians(args.ply))
    xyz = np.asarray(g["xyz"], dtype=np.float64)
    col = np.asarray(g["color"], dtype=np.float32)
    print(f"[info] {args.ply}: {len(xyz):,} Gaussians")

    f, cx, cy = D.intrinsic(args.size, args.fovy)
    cams = orbit_cams(args.views, args.elev_lo, args.elev_hi, args.radius, args.size, args.fovy)

    from PIL import Image

    entries = []
    for i, (R, eye, azim, elev) in enumerate(cams):
        depth, rgb = D.render_depth(xyz, col, size=args.size, fovy=args.fovy, R=R, eye=eye)
        alpha = (depth > 0).astype(np.uint8) * 255
        rgba = np.dstack([rgb, alpha])
        rel = f"views/{i:03d}.png"
        Image.fromarray(rgba).save(os.path.join(args.out, rel))

        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = -R @ eye
        entries.append(
            {
                "mask": rel,
                "width": args.size,
                "height": args.size,
                "fx": float(f),
                "fy": float(f),
                "cx": float(cx),
                "cy": float(cy),
                "world_to_cam": w2c.tolist(),
                "azim_deg": float(azim),
                "elev_deg": float(elev),
            }
        )
        print(
            f"[view {i:02d}] azim {azim:6.1f}° elev {elev:6.1f}° "
            f"Maske {alpha.mean() / 255 * 100:5.1f}% belegt"
        )

    meta = {
        "generator": "dryrun-from-splat",
        "source": os.path.abspath(args.ply),
        "size": args.size,
        "fovy": args.fovy,
        "views": entries,
    }
    with open(os.path.join(args.out, "views.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[out] {os.path.join(args.out, 'views.json')}")


if __name__ == "__main__":
    main()
