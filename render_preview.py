"""Minimaler Gaussian-Splat-Renderer (reines PyTorch, kein CUDA).

Ersatz fuer das Vorschau-Video von LGM: upstream rendert die .ply mit
`diff-gaussian-rasterization`, einer CUDA-C++-Extension, die auf Intel-iGPUs
nicht existiert. Dieses Skript rendert stattdessen eine Turntable-Animation
rein vektoriell auf CPU (oder XPU).

    .venv\\Scripts\\python.exe render_preview.py --ply LGM\\workspace_test\\catstatue_rgba.ply

Nicht differentierbar und ohne echtes Tiefensortieren -- gedacht als schnelle
Sichtkontrolle, nicht als Ersatz fuer einen echten Splat-Viewer.
"""

import argparse
import os

import numpy as np
import torch

# --- PLY (3DGS-Standardformat von LGM) laden -------------------------------


def load_ply(path):
    from plyfile import PlyData

    data = PlyData.read(path).elements[0].data
    device = torch.device("cpu")

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    # LGM speichert bereits aktivierte Werte rueckgerechnet -> rueckgaengig machen
    opacity = np.asarray(data["opacity"], dtype=np.float32)[:, None]
    scale = np.stack([data["scale_0"], data["scale_1"], data["scale_2"]], axis=1).astype(np.float32)
    rot = np.stack([data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]], axis=1).astype(np.float32)
    f_dc = np.stack([data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]], axis=1).astype(np.float32)

    c = {}
    c["xyz"] = torch.from_numpy(xyz).to(device)
    c["opacity"] = torch.sigmoid(torch.from_numpy(opacity))
    c["scale"] = torch.exp(torch.from_numpy(scale))
    c["quat"] = torch.from_numpy(rot)  # (w, x, y, z), bereits normalisiert
    c["color"] = (0.28209479177387814 * torch.from_numpy(f_dc) + 0.5).clamp(0, 1)
    return c


# --- Kamera ---------------------------------------------------------------


def look_at(eye, target, up=torch.tensor([0.0, 1.0, 0.0])):
    """OpenCV-Konvention: Kamera schaut entlang +z, y zeigt nach unten."""
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.cross(forward, up, dim=0), dim=0)
    true_up = torch.cross(right, forward, dim=0)
    return torch.stack([right, true_up, forward], dim=0)  # [3,3]


def orbit(elev_deg, azim_deg, radius):
    el = np.deg2rad(elev_deg)
    az = np.deg2rad(azim_deg)
    eye = torch.tensor(
        [np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)],
        dtype=torch.float32,
    ) * radius
    return look_at(eye, torch.zeros(3)), eye


# --- Kernalgorithmus ------------------------------------------------------


def quat_to_rotmat(q):
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )


def render(c, R_cam, eye, size, fovy_deg, bg=(1.0, 1.0, 1.0)):
    """c: dict mit xyz/opacity/scale/quat/color. R_cam: [3,3] world->camera, eye: [3]."""
    W = H = size
    fy = (H / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy

    # Weltraum -> Kameraraum (Translation nicht vergessen!)
    pc = (c["xyz"] - eye) @ R_cam.T  # [N,3]
    z = pc[:, 2]
    valid = z > 0.2
    pc, z = pc[valid], z[valid]
    idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    opa, sca, rot, col = c["opacity"][idx], c["scale"][idx], c["quat"][idx], c["color"][idx]

    # 3D-Kovarianz im Kameraraum: R_world->cam * (R_q diag(s^2) R_q^T) * R^T
    Rq = quat_to_rotmat(rot)  # [N,3,3]
    S = torch.diag_embed(sca**2)  # [N,3,3]
    M = R_cam.unsqueeze(0) @ Rq  # [N,3,3]
    cov_cam = M @ S @ M.transpose(1, 2)

    # Jacobi-Projektion auf 2D (Bildkoordinaten wachsen nach UNTEN -> v gespiegelt)
    x, y = pc[:, 0], pc[:, 1]
    J = torch.zeros(len(z), 2, 3)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z**2
    J[:, 1, 1] = -fy / z
    J[:, 1, 2] = fy * y / z**2
    cov2d = J @ cov_cam @ J.transpose(1, 2)  # [N,2,2]
    cov2d[:, 0, 0] += 0.3  # Anti-Aliasing-Epsilon
    cov2d[:, 1, 1] += 0.3

    det = cov2d[:, 0, 0] * cov2d[:, 1, 1] - cov2d[:, 0, 1] ** 2
    ok = det > 1e-9
    inv = torch.zeros_like(cov2d)
    inv[:, 0, 0] = cov2d[:, 1, 1] / det
    inv[:, 1, 1] = cov2d[:, 0, 0] / det
    inv[:, 0, 1] = -cov2d[:, 0, 1] / det
    inv[:, 1, 0] = -cov2d[:, 1, 0] / det

    # Bildschirmkoordinaten (y nach unten)
    u = fx * x / z + W / 2.0
    v = -fy * y / z + H / 2.0

    # Radius aus groesstem Eigenwert
    tr = cov2d[:, 0, 0] + cov2d[:, 1, 1]
    disc = torch.sqrt(torch.clamp((cov2d[:, 0, 0] - cov2d[:, 1, 1]) ** 2 / 4 + cov2d[:, 0, 1] ** 2, min=0))
    lam = tr / 2 + disc
    # KEIN hartes Clamping mehr: ein gestauchter Splat aendert seine scheinbare
    # Groesse mit dem Blickwinkel, was beim Drehen wie wandernde Proportionen wirkt.
    rad = torch.ceil(3.0 * torch.sqrt(torch.clamp(lam, min=1e-9))).clamp(1, None)

    keep = ok & (u > -rad) & (u < W + rad) & (v > -rad) & (v < H + rad) & (opa.squeeze(-1) > 0.004)
    if keep.sum() == 0:
        return torch.tensor(bg).view(1, 3).expand(H * W, 3).reshape(H, W, 3).numpy()

    u, v, rad, inv = u[keep], v[keep], rad[keep], inv[keep]
    opa, col, z = opa[keep], col[keep], z[keep]

    # Von hinten nach vorne sortieren -> Alpha-Over statt Farbmittelung.
    order = torch.argsort(z, descending=True)
    u, v, rad, inv, opa, col, z = u[order], v[order], rad[order], inv[order], opa[order], col[order], z[order]
    rad_i = rad.long()

    acc_c = torch.zeros(H * W, 3)
    acc_w = torch.zeros(H * W, 1)
    bg_t = torch.tensor(bg, dtype=torch.float32).view(1, 3)

    # Chunks werden dynamisch geschnitten: das Offset-Raster richtet sich nach
    # dem groessten Radius *im* Chunk, damit auch grosse Splats ungestaucht
    # bleiben, ohne dass der Speicher explodiert.
    ENTRY_BUDGET = 400_000
    CH_MAX = 64          # zu grosse Tiefen-Chunks erzeugen sichtbare Konturen
    i, N = 0, u.shape[0]
    while i < N:
        rmax = int(rad_i[i].item())
        j = i + 1
        while j < N:
            if (j - i) >= CH_MAX:
                break
            nr = max(rmax, int(rad_i[j].item()))
            if (2 * nr + 1) ** 2 * (j - i + 1) > ENTRY_BUDGET:
                break
            rmax = nr
            j += 1

        offs = torch.arange(-rmax, rmax + 1)
        gy, gx = torch.meshgrid(offs, offs, indexing="ij")
        oy, ox = gy.reshape(1, -1), gx.reshape(1, -1)  # [1, K]

        uu, vv, rr = u[i:j], v[i:j], rad[i:j]
        px = (uu[:, None] + ox).round().long()
        py = (vv[:, None] + oy).round().long()

        dx = (px - uu[:, None]).float()
        dy = (py - vv[:, None]).float()
        d2 = dx * dx + dy * dy

        # kreisrunde Begrenzung exakt am eigenen 3-sigma-Radius
        inside = (d2 <= (rr[:, None] ** 2)) & (px >= 0) & (px < W) & (py >= 0) & (py < H)

        i00 = inv[i:j, 0, 0][:, None]
        i01 = inv[i:j, 0, 1][:, None]
        i11 = inv[i:j, 1, 1][:, None]
        power = 0.5 * (i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy)
        w = torch.exp(-power) * opa[i:j]  # [B,K,1]

        # weiches Fenster: am Maskenrand geht das Gewicht gegen 0, sonst Ringe
        q2 = (d2 / (rr[:, None] ** 2)).clamp(0, 1)
        w = w * (1 - q2) ** 2

        pix = (py * W + px)[inside]
        ww = w.reshape(-1)[inside.reshape(-1)]
        cc = col[i:j].unsqueeze(1).expand(-1, ox.shape[1], -1).reshape(-1, 3)[inside.reshape(-1)]

        # Transmissionsgrad am Chunk-Anfang: was schon davor liegt, verdeckt
        T = (1.0 - acc_w).clamp(0, 1)
        wgt = ww * T[pix, 0]
        acc_c.index_add_(0, pix, cc * wgt[:, None])
        acc_w.index_add_(0, pix, wgt[:, None])

        i = j

    alpha = acc_w.clamp(0, 1)
    img = acc_c + bg_t * (1 - alpha)
    return img.reshape(H, W, 3).clamp(0, 1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--out", default=None, help="Zielordner fuer png/mp4")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--frames", type=int, default=0, help="0 = Einzelbild, >0 = Turntable")
    ap.add_argument("--radius", type=float, default=0.0, help="0 = automatisch aus der Groesse ableiten")
    ap.add_argument("--fovy", type=float, default=49.1)
    ap.add_argument("--elev", type=float, default=10.0)
    args = ap.parse_args()

    c = load_ply(args.ply)
    print(f"[INFO] {args.ply}: {c['xyz'].shape[0]} Gaussians")

    if args.radius <= 0:
        # 99%-Perzentil der Punktwolken-Ausdehnung -> Objekt fuellt das Bild
        extent = torch.quantile(c["xyz"].norm(dim=1), 0.99).item()
        args.radius = max(1.0, 2.7 * extent)
        print(f"[INFO] radius automatisch: {args.radius:.2f} (Ausdehnung {extent:.2f})")

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ply))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.ply))[0]

    if args.frames <= 0:
        R, eye = orbit(args.elev, 0, args.radius)
        img = render(c, R, eye, args.size, args.fovy)
        path = os.path.join(out_dir, stem + "_preview.png")
        from PIL import Image

        Image.fromarray((img * 255).astype(np.uint8)).save(path)
        print(f"[INFO] -> {path}")
        return

    import imageio.v2 as imageio
    from PIL import Image

    frames = []
    for i in range(args.frames):
        R, eye = orbit(args.elev, 360.0 * i / args.frames, args.radius)
        frames.append((render(c, R, eye, args.size, args.fovy) * 255).astype(np.uint8))
        if (i + 1) % 10 == 0:
            print(f"[INFO] {i + 1}/{args.frames}")

    mp4 = os.path.join(out_dir, stem + "_turntable.mp4")
    imageio.mimwrite(mp4, frames, fps=24, quality=8)
    gif = os.path.join(out_dir, stem + "_turntable.gif")
    Image.fromarray(frames[len(frames) // 4]).save(gif, save_all=True, append_images=[
        Image.fromarray(f) for f in frames[len(frames) // 4 + 1:] + frames[: len(frames) // 4]
    ], duration=42, loop=0)
    print(f"[INFO] -> {mp4}\n[INFO] -> {gif}")


if __name__ == "__main__":
    main()
