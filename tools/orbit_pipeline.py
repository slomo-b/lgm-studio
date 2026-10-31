"""Naht zwischen Multi-View-Generator und 3DGS-Training.

Nimmt einen Ordner mit `views.json` (splatcage-Format) und macht daraus:

  1. eine Voxelhuelle  -> hull.bin          (splatcage, Rust)
  2. eine Startpunktwolke -> hull_points.ply (Voxelzentren mit Farbe aus den Views)
  3. ein 3DGS-Dataset  -> transforms.json    (nerfstudio-Konvention, c2w OpenGL)

Die Huelle ist hier nicht nur Diagnose: sie liefert die *Initialisierung*.
Standard-3DGS startet aus COLMAP-Punkten; wir haben kein COLMAP, aber eine
Huelle, und die ist als Startverteilung sogar besser -- sie enthaelt per
Konstruktion keinen Punkt im Leerraum.

    .venv-simready\\Scripts\\python.exe tools\\orbit_pipeline.py --dir out\\orbit_test
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def solve_4x4(m):
    return np.linalg.inv(np.asarray(m, dtype=np.float64))


def read_obj_points(path):
    pts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                pts.append((float(x), float(y), float(z)))
    return np.asarray(pts, dtype=np.float64)


def sample_colors(points, views):
    """Fuer jeden Punkt die Farbe aus der Ansicht, die ihn am direktesten sieht."""
    n = len(points)
    colors = np.full((n, 3), 0.6, dtype=np.float32)

    eyes = []
    fwds = []
    for v in views:
        w2c = np.asarray(v["world_to_cam"], dtype=np.float64)
        R, t = w2c[:3, :3], w2c[:3, 3]
        eye = -R.T @ t
        fwd = R[2]  # dritte Zeile = Kamerarichtung in Weltkoordinaten
        eyes.append(eye)
        fwds.append(fwd / np.linalg.norm(fwd))
    eyes = np.stack(eyes)
    fwds = np.stack(fwds)

    # beste Ansicht pro Punkt (Vektor zum Punkt vs. Blickrichtung)
    dirs = points[:, None, :] - eyes[None, :, :]
    dirs /= np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-9)
    score = np.einsum("nvk,vk->nv", dirs, fwds)
    best = np.argmax(score, axis=1)

    try:
        from PIL import Image
    except Exception:
        return colors

    cache = {}
    for i, v in enumerate(views):
        sel = np.flatnonzero(best == i)
        if sel.size == 0:
            continue
        img_path = v.get("_rgb_path")
        if not img_path or not os.path.exists(img_path):
            continue
        if i not in cache:
            cache[i] = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        img = cache[i]
        h, w = img.shape[:2]
        w2c = np.asarray(v["world_to_cam"], dtype=np.float64)
        p = points[sel]
        pc = p @ w2c[:3, :3].T + w2c[:3, 3]
        z = pc[:, 2]
        ok = z > 1e-6
        u = np.round(v["fx"] * pc[:, 0] / np.maximum(z, 1e-9) + v["cx"]).astype(int)
        vv = np.round(v["fy"] * pc[:, 1] / np.maximum(z, 1e-9) + v["cy"]).astype(int)
        ok &= (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        idx = sel[ok]
        colors[idx] = img[vv[ok], u[ok]]
    return colors


def write_points_ply(points, colors, path):
    """3DGS-taugliche Punktwolke: x,y,z plus RGB als uchar."""
    n = len(points)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    data = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    data["x"], data["y"], data["z"] = points[:, 0], points[:, 1], points[:, 2]
    rgb = np.clip(colors, 0, 1) * 255
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def cv_to_gl_c2w(w2c):
    """world_to_cam (OpenCV) -> camera_to_world (OpenGL), wie nerfstudio es will."""
    c2w = np.linalg.inv(np.asarray(w2c, dtype=np.float64))
    c2w[:3, 1:3] *= -1.0  # y und z spiegeln
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Ordner mit views.json und views/")
    ap.add_argument(
        "--splatcage",
        default=os.path.join("splatcage", "target", "release", "splatcage.exe"),
    )
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--margin", type=int, default=1)
    ap.add_argument("--threshold", type=int, default=200)
    args = ap.parse_args()

    root = args.dir
    with open(os.path.join(root, "views.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    views = meta["views"]
    for v in views:
        v["_rgb_path"] = os.path.join(root, v["mask"])
    print(f"[info] {len(views)} Ansichten aus {root}")

    hull_path = os.path.join(root, "hull.bin")
    obj_path = os.path.join(root, "hull.obj")
    cmd = [
        args.splatcage,
        "build",
        "--views",
        os.path.abspath(os.path.join(root, "views.json")),
        "--res",
        str(args.res),
        "--margin",
        str(args.margin),
        "--mask-mode",
        "alpha",
        "--threshold",
        str(args.threshold),
        "--out",
        os.path.abspath(hull_path),
        "--obj",
        os.path.abspath(obj_path),
    ]
    print(f"[run] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"splatcage build fehlgeschlagen ({proc.returncode})")

    pts = read_obj_points(obj_path)
    print(f"[info] Huelle: {len(pts):,} Punkte")

    colors = sample_colors(pts, views)
    ply_path = os.path.join(root, "hull_points.ply")
    write_points_ply(pts, colors, ply_path)
    print(f"[out] {ply_path}")

    # nerfstudio / 3DGS
    frames = []
    for i, v in enumerate(views):
        c2w = cv_to_gl_c2w(v["world_to_cam"])
        frames.append(
            {
                "file_path": f"views/{i:03d}.png",
                "transform_matrix": c2w.tolist(),
            }
        )
    ns = {
        "camera_model": "OPENCV",
        "fl_x": views[0]["fx"],
        "fl_y": views[0]["fy"],
        "cx": views[0]["cx"],
        "cy": views[0]["cy"],
        "w": views[0]["width"],
        "h": views[0]["height"],
        "frames": frames,
    }
    ns_path = os.path.join(root, "transforms.json")
    with open(ns_path, "w", encoding="utf-8") as f:
        json.dump(ns, f, indent=2)
    print(f"[out] {ns_path}")

    report = {
        "views": len(views),
        "hull_points": int(len(pts)),
        "hull_file": hull_path,
        "init_ply": ply_path,
        "nerfstudio": ns_path,
        "res": args.res,
    }
    with open(os.path.join(root, "orbit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[done] {len(views)} Ansichten, {len(pts):,} Init-Punkte")


if __name__ == "__main__":
    main()
