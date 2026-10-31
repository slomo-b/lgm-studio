"""Kalibriert die Kamera-Konvention gegen die generierten Ansichten.

Problem: die Posen, die ich aus `get_camera` abgeleitet habe, passen nicht --
nur 1,9 % der Huelle landen im Bild. Statt zu raten, werden die plausiblen
Konventionen durchprobiert und nach Huellenqualitaet bewertet:

  Anteil der groessten Komponente  hoch  = zusammenhaengendes Objekt
  Belegung                         niedrig = scharfe Huelle

Eine falsche Pose schneidet quer und hinterlaesst entweder fast nichts
(leere Huelle) oder viele Splitter (niedriger Komponentenanteil).

    .venv-simready\\Scripts\\python.exe tools\\calibrate_pose.py --dir out\\elev_views
"""

import argparse
import itertools
import json
import os
import re
import subprocess

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


def build(meta, path, cage, out_bin):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    proc = subprocess.run(
        [
            cage, "build", "--views", os.path.abspath(path),
            "--res", "160", "--margin", "1",
            "--mask-mode", "luma", "--threshold", "240",
            "--out", os.path.abspath(out_bin),
        ],
        capture_output=True, text=True,
    )
    text = proc.stdout
    occ = re.search(r"([\d.]+)% belegt, (\d+) ms", text)
    lar = re.search(r"\[largest\] (\d+) -> (\d+) Voxel, ([\d.]+)%", text)
    if not lar:
        return None
    before, after, pct = int(lar.group(1)), int(lar.group(2)), float(lar.group(3))
    return {
        "occupancy": pct,
        "share": after / max(before, 1),
        "voxels": after,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="out/elev_views")
    ap.add_argument("--cage", default="splatcage/target/release/splatcage.exe")
    ap.add_argument("--tag", default="e0", help="Bündel zum Kalibrieren")
    args = ap.parse_args()

    meta = json.load(open(f"{args.dir}/views.json", encoding="utf-8"))
    views = [v for v in meta["views"] if v["elev_deg"] == meta["views"][0]["elev_deg"]]
    if args.tag.startswith("e0"):
        views = [v for v in meta["views"] if v["elev_deg"] == 0.0]
    print(f"[info] {len(views)} Ansichten zum Kalibrieren")

    rows = []
    for off, sign, radius in itertools.product((0, 90, 180, 270), (1, -1), (1.1, 1.5, 2.0)):
        test = {"views": []}
        for i, v in enumerate(views):
            w = dict(v)
            w["world_to_cam"] = w2c(off + 90.0 * i, sign * v["elev_deg"], radius).tolist()
            test["views"].append(w)
        r = build(test, os.path.join(args.dir, "_cal.json"), args.cage,
                  os.path.join(args.dir, "_cal.bin"))
        if r:
            rows.append((off, sign, radius, r))

    rows.sort(key=lambda t: (-t[3]["share"], t[3]["occupancy"]))
    print(f"{'offset':>7} {'elev':>5} {'radius':>7} {'belegt':>9} {'komponente':>11}")
    print("-" * 44)
    for off, sign, radius, r in rows[:12]:
        print(
            f"{off:>7} {sign*1:>+5} {radius:>7.1f} {r['occupancy']:>8.2f}% {r['share']*100:>10.1f}%"
        )
    best = rows[0]
    print(f"\n[beste] offset={best[0]} elev_sign={best[1]:+d} radius={best[2]}")


if __name__ == "__main__":
    main()
