"""Schreibt views.json mit korrigierter Kamera-Konvention neu.

Getestet wird das Elevationsvorzeichen: wenn das Modell +25 Grad als Blick von
unten interpretiert (oder umgekehrt), paaren sich die Bilder mit der falschen
Kamera und die Huelle schneidet sich selbst weg.

    .venv-simready\\Scripts\\python.exe tools\\rewrite_poses.py ^
        --in out\\elev_views\\views.json --elev-sign -1 --out out\\elev_views\\views_flip.json
"""

import argparse
import json

import numpy as np


def w2c(azim_deg, elev_deg, radius):
    a, e = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    eye = radius * np.array(
        [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)], dtype=np.float64
    )
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = -R @ eye
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--elev-sign", type=int, default=1, choices=[1, -1])
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--azim-offset", type=float, default=0.0)
    args = ap.parse_args()

    meta = json.load(open(args.src, encoding="utf-8"))
    for v in meta["views"]:
        azim = v["azim_deg"] + args.azim_offset
        elev = args.elev_sign * v["elev_deg"]
        v["world_to_cam"] = w2c(azim, elev, args.radius).tolist()
        v["used_azim"] = azim
        v["used_elev"] = elev
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[out] {args.out}: {len(meta['views'])} Views, elev-sign {args.elev_sign:+d}")


if __name__ == "__main__":
    main()
