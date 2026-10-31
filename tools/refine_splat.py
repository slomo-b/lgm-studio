"""Feinschliff: 3DGS gegen die vier LGM-Ansichten optimieren -- ohne CUDA.

Das ist der Hebel "predict-then-refine" aus LGM_HEBEL.md, nur ohne den
CUDA-Rasterizer: `simready/diffrender.py` rendert differenzierbar auf der Arc.

Was optimiert wird: Position, Groesse, Deckkraft und Farbe jeder Gaussian.
Rotation bleibt fest (sie ist bei LGM bereits plausibel und kostet nur Zeit).

Gemessen auf der Arc-iGPU: ~2.5 s pro Schritt (vorwaerts + rueckwaerts), also
150 Schritte ~6 min pro Asset.

    .venv\\Scripts\\python.exe tools\\refine_splat.py --name catstatue_rgba --steps 150
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiui.cam import orbit_camera  # noqa: E402
from plyfile import PlyData, PlyElement  # noqa: E402

from simready import diffrender as DR  # noqa: E402
from simready import splat as S  # noqa: E402

SH_C0 = 0.28209479177387814
FIELDS = [
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]


def cam(azimuth, radius=1.5, elevation=0.0):
    pose = orbit_camera(elevation, azimuth, radius=radius, opengl=True)
    eye = torch.tensor(pose[:3, 3], dtype=torch.float32)
    right = torch.tensor(pose[:3, 0], dtype=torch.float32)
    up = torch.tensor(pose[:3, 1], dtype=torch.float32)
    back = torch.tensor(pose[:3, 2], dtype=torch.float32)
    return torch.stack([right, -up, -back], dim=0), eye


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-9 else 10 * np.log10(255.0**2 / mse)


def to_uint8(img):
    return (img.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_ply(path, xyz, scale, quat, opacity, color):
    """Schreibt ein LGM-kompatibles .ply (SH-Grad 0)."""
    n = len(xyz)
    arr = np.empty(n, dtype=[(f, "f4") for f in FIELDS])
    xyz = xyz.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    quat = quat.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    color = color.detach().cpu().numpy()

    arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    f_dc = (np.clip(color, 0.0, 1.0) - 0.5) / SH_C0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = np.log(np.clip(opacity, 1e-4, 1 - 1e-4) / (1 - np.clip(opacity, 1e-4, 1 - 1e-4)))
    ls = np.log(np.clip(scale, 1e-6, None))
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = ls[:, 0], ls[:, 1], ls[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3],
    )
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="catstatue_rgba")
    ap.add_argument("--ply", default=None)
    ap.add_argument("--workspace", default="LGM/workspace_test")
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr-xyz", type=float, default=1.5e-3)
    ap.add_argument("--lr-scale", type=float, default=3e-3)
    ap.add_argument("--lr-opacity", type=float, default=2e-2)
    ap.add_argument("--lr-color", type=float, default=2e-2)
    ap.add_argument(
        "--reg-shift",
        type=float,
        default=0.1,
        help="Strafe fuer Bewegung weg vom Ausgangspunkt; bremst Sprenkel in Zwischenansichten",
    )
    ap.add_argument(
        "--reg-color",
        type=float,
        default=0.05,
        help="Strafe fuer extreme Farbaenderungen; verhindert einzelne grelle Gaussians",
    )
    ap.add_argument("--out", default="out/refine")
    args = ap.parse_args()

    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    src = args.ply or os.path.join(args.workspace, args.name + ".ply")
    g = S.clean(S.load_gaussians(src))
    n = len(g["xyz"])
    print(f"[info] {src}: {n:,} Gaussians, Device {dev}")

    # Ziele und Kameras
    targets, cams = [], []
    for i in range(4):
        p = os.path.join(args.workspace, "mv", f"{args.name}_view{i}.png")
        im = Image.open(p).convert("RGB").resize((args.size, args.size), Image.BILINEAR)
        targets.append(torch.tensor(np.asarray(im), dtype=torch.float32, device=dev) / 255.0)
        R, eye = cam(90.0 * i)
        cams.append((R.to(dev), eye.to(dev)))

    # Rohe Parameter (unbeschraenkt) -- erst am Ende aktivieren
    xyz = torch.tensor(g["xyz"], dtype=torch.float32, device=dev, requires_grad=True)
    xyz0 = xyz.detach().clone()
    log_s = torch.tensor(np.log(np.clip(g["scale"], 1e-6, None)), dtype=torch.float32,
                         device=dev, requires_grad=True)
    op_logit = torch.tensor(
        np.log(np.clip(g["opacity"], 1e-4, 1 - 1e-4) / (1 - np.clip(g["opacity"], 1e-4, 1 - 1e-4))),
        dtype=torch.float32, device=dev, requires_grad=True,
    )
    col_logit = torch.tensor(
        np.log(np.clip(g["color"], 1e-4, 1 - 1e-4) / (1 - np.clip(g["color"], 1e-4, 1 - 1e-4))),
        dtype=torch.float32, device=dev, requires_grad=True,
    )
    col0 = col_logit.detach().clone()
    quat = torch.tensor(g["rot"], dtype=torch.float32, device=dev)
    quat = torch.nn.functional.normalize(quat, dim=-1)

    opt = torch.optim.Adam(
        [
            {"params": [xyz], "lr": args.lr_xyz},
            {"params": [log_s], "lr": args.lr_scale},
            {"params": [op_logit], "lr": args.lr_opacity},
            {"params": [col_logit], "lr": args.lr_color},
        ]
    )

    def render_view(i, detach=True):
        R, eye = cams[i]
        s = torch.exp(log_s) if detach else torch.exp(log_s)
        a = torch.sigmoid(op_logit)
        c = torch.sigmoid(col_logit)
        img, _ = DR.render(xyz, s, quat, a, c, R, eye, size=args.size)
        return img

    def evaluate(tag):
        scores = []
        for i in range(4):
            with torch.no_grad():
                out = to_uint8(render_view(i))
            scores.append(psnr(out, np.asarray(targets[i].cpu()) * 255))
        print(f"[{tag}] PSNR " + " ".join(f"{s:5.1f}" for s in scores)
              + f" | Mittel {np.mean(scores):5.2f}", flush=True)
        return float(np.mean(scores))

    before = evaluate("vorher")

    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        i = (step - 1) % 4
        img = render_view(i, detach=False)
        loss = (img - targets[i]).abs().mean()
        if args.reg_shift > 0:
            loss = loss + args.reg_shift * ((xyz - xyz0) ** 2).sum(1).mean()
        if args.reg_color > 0:
            loss = loss + args.reg_color * ((col_logit - col0) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 25 == 0 or step == args.steps:
            dt = time.perf_counter() - t_start
            print(f"[{step:4d}/{args.steps}] loss {loss.item():.4f} "
                  f"({dt / step:.2f} s/Schritt)", flush=True)

    after = evaluate("nachher")
    print(f"[ok] {before:.2f} -> {after:.2f} dB ({after - before:+.2f})")

    os.makedirs(args.out, exist_ok=True)
    out_ply = os.path.join(args.out, args.name + "_refined.ply")
    with torch.no_grad():
        save_ply(out_ply, xyz, torch.exp(log_s), quat, torch.sigmoid(op_logit),
                 torch.sigmoid(col_logit))
    print(f"[out] {out_ply}")

    # Vergleich aus einer Zwischenansicht (nicht optimiert -> zeigt, ob es generalisiert)
    tiles = []
    for side, src_ply in (("vorher", src), ("nachher", out_ply)):
        gg = S.clean(S.load_gaussians(src_ply))
        m = torch.tensor(gg["xyz"], dtype=torch.float32, device=dev)
        sc = torch.tensor(gg["scale"], dtype=torch.float32, device=dev)
        q = torch.tensor(gg["rot"], dtype=torch.float32, device=dev)
        o = torch.tensor(gg["opacity"], dtype=torch.float32, device=dev)
        c = torch.tensor(gg["color"], dtype=torch.float32, device=dev)
        R, eye = cam(45.0)
        with torch.no_grad():
            img, _ = DR.render(m, sc, q, o, c, R.to(dev), eye.to(dev), size=args.size)
        tiles.append(to_uint8(img))
    strip = np.concatenate(tiles, axis=1)
    p = os.path.join(args.out, args.name + "_neu45.png")
    Image.fromarray(strip).save(p)
    print(f"[out] {p}  (links vorher, rechts nachher, Azimut 45 Grad)")


if __name__ == "__main__":
    main()
