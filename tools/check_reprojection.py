"""Diagnose: passen die angenommenen Posen zu den Masken?

Fuer jeden Punkt der Huelle wird geprueft, ob er in jeder Ansicht auf ein
Objektpixel faellt. Eine Ansicht mit niedriger Trefferquote hat entweder eine
falsche Pose oder zeigt ein anderes Objekt -- beides macht Silhouetten-Carving
unbrauchbar.
"""

import json
import sys

import numpy as np
from PIL import Image


def load_points(path):
    return np.array(
        [[float(x) for x in line.split()[1:4]] for line in open(path) if line.startswith("v ")]
    )


def agreement(points, view, root):
    img = np.asarray(Image.open(f"{root}/{view['mask']}").convert("L"), dtype=np.float32)
    w2c = np.asarray(view["world_to_cam"])
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    ok = z > 1e-6
    u = np.round(view["fx"] * pc[:, 0] / np.maximum(z, 1e-9) + view["cx"]).astype(int)
    v = np.round(view["fy"] * pc[:, 1] / np.maximum(z, 1e-9) + view["cy"]).astype(int)
    ok &= (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
    inside = np.zeros(len(points), dtype=bool)
    inside[ok] = img[v[ok], u[ok]] < 240
    return float(inside.mean()), int(ok.sum())


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "out/elev_views"
    tags = sys.argv[2:] or ["e0", "em25", "ep25"]
    for tag in tags:
        meta_path = f"{root}/views_{tag}.json"
        obj_path = f"{root}/hull_{tag}.obj"
        try:
            meta = json.load(open(meta_path, encoding="utf-8"))
            pts = load_points(obj_path)
        except FileNotFoundError:
            continue
        print(f"--- {tag}: {len(pts):,} Punkte ---")
        for i, view in enumerate(meta["views"]):
            frac, projected = agreement(pts, view, root)
            print(
                f"  view{i}  az {view['azim_deg']:5.0f}  el {view['elev_deg']:+5.0f}"
                f"   im Bild {projected / len(pts) * 100:5.1f}%"
                f"   auf Objekt {frac * 100:5.1f}%"
            )


if __name__ == "__main__":
    main()
