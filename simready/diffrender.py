"""Differenzierbarer Splat-Renderer ohne CUDA — exaktes Front-to-Back.

Warum nicht einfach akkumulieren: normierte Summen mitteln alle Gaussians entlang
eines Strahls, das Bild wird milchig (gemessen). Echte Verdeckung braucht eine
Reihenfolge pro Pixel.

Der uebliche Weg (Schleife ueber Splats, pro Pixel Transmission nachfuehren)
braucht Index-Operationen, die auf dieser XPU-Plattform haengen (`index_add_`).
Deshalb hier ein vektorisierter Ersatz:

  1. Alle Beitraege (Pixel, Tiefe, Gewicht, Farbe) einsammeln
  2. Nach (Pixel, Tiefe) sortieren
  3. Transmission als Segment-Produkt:  T_i = cumprod(1-a)[i] / cumprod(1-a)[start_i]
  4. Beitrag_i = Farbe_i * a_i * T_i, per `scatter_add` aufs Bild

Das ist numerisch dasselbe wie die klassische Formel, braucht nur
`argsort`, `cumprod`, `searchsorted`, `gather` und `scatter_add` — alles
gemessen verfuegbar (`tools/xpu_opcheck.py`).
"""

import numpy as np
import torch


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


def _project(means, scales, quats, R, eye, f, cx, cy, min_z=0.05):
    pc = (means - eye) @ R.transpose(0, 1)
    z = pc[:, 2]
    zc = z.clamp(min=min_z)
    x, y = pc[:, 0], pc[:, 1]

    u = f * x / zc + cx
    v = f * y / zc + cy  # gleiche Konvention wie simready/depth.py

    Rq = quat_to_rotmat(quats)
    M = R.unsqueeze(0) @ Rq
    S2 = torch.diag_embed(scales * scales)
    cov_cam = M @ S2 @ M.transpose(1, 2)

    J = torch.zeros(len(z), 2, 3, device=means.device, dtype=means.dtype)
    J[:, 0, 0] = f / zc
    J[:, 0, 2] = -f * x / (zc * zc)
    J[:, 1, 1] = -f / zc
    J[:, 1, 2] = f * y / (zc * zc)
    cov2d = J @ cov_cam @ J.transpose(1, 2)

    a = cov2d[:, 0, 0] + 0.3
    d = cov2d[:, 1, 1] + 0.3
    b = cov2d[:, 0, 1]
    det = (a * d - b * b).clamp(min=1e-9)
    inv = torch.zeros_like(cov2d)
    inv[:, 0, 0] = d / det
    inv[:, 1, 1] = a / det
    inv[:, 0, 1] = -b / det
    inv[:, 1, 0] = -b / det

    lam = (a + d) / 2 + torch.sqrt(torch.clamp((a - d) ** 2 / 4 + b * b, min=0))
    return u, v, z, inv, lam, (z > min_z)


def render(
    means,
    scales,
    quats,
    opacity,
    colors,
    R,
    eye,
    size=256,
    fovy=49.1,
    bg=(1.0, 1.0, 1.0),
    max_radius=16,
    chunk=8192,
):
    """Rendert eine Ansicht; alle Parameter duerfen Blatt-Tensoren sein."""
    H = W = size
    f = (H * 0.5) / np.tan(np.deg2rad(fovy) * 0.5)
    cx = cy = H * 0.5
    dev, dt = means.device, means.dtype

    u, v, z, inv, lam, ok = _project(means, scales, quats, R, eye, f, cx, cy)
    rad = torch.ceil(3.0 * torch.sqrt(lam.clamp(min=1e-9))).clamp(1, max_radius)

    keep = ok & (u > -rad) & (u < W + rad) & (v > -rad) & (v < H + rad)
    sel = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if sel.numel() == 0:
        return (
            torch.tensor(bg, device=dev, dtype=dt).view(1, 3).expand(H * W, 3).reshape(H, W, 3),
            torch.zeros(H, W, 1, device=dev, dtype=dt),
        )

    u = u.index_select(0, sel)
    v = v.index_select(0, sel)
    z = z.index_select(0, sel)
    rad = rad.index_select(0, sel)
    inv = inv.index_select(0, sel)
    opa = opacity.index_select(0, sel)
    col = colors.index_select(0, sel)

    order = torch.argsort(rad)
    u, v, z = u.index_select(0, order), v.index_select(0, order), z.index_select(0, order)
    rad, inv = rad.index_select(0, order), inv.index_select(0, order)
    opa, col = opa.index_select(0, order), col.index_select(0, order)

    pix_l, a_l, c_l, z_l = [], [], [], []
    for s in range(0, len(u), chunk):
        sl = slice(s, s + chunk)
        r = rad[sl]
        rmax = int(r.max().item())
        offs = torch.arange(-rmax, rmax + 1, device=dev, dtype=dt)
        gy, gx = torch.meshgrid(offs, offs, indexing="ij")
        ox, oy = gx.reshape(1, -1), gy.reshape(1, -1)

        uu, vv = u[sl].unsqueeze(1), v[sl].unsqueeze(1)
        rr = r.unsqueeze(1).to(dt)
        px = (uu + ox).round()
        py = (vv + oy).round()
        dx, dy = px - uu, py - vv

        i00, i01, i11 = inv[sl, 0, 0].unsqueeze(1), inv[sl, 0, 1].unsqueeze(1), inv[sl, 1, 1].unsqueeze(1)
        power = 0.5 * (i00 * dx * dx + 2 * i01 * dx * dy + i11 * dy * dy)
        w = torch.exp(-power) * opa[sl].unsqueeze(1)

        inside = (
            (dx * dx + dy * dy <= rr * rr)
            & (px >= 0) & (px < W) & (py >= 0) & (py < H)
        ).reshape(-1)
        s2 = torch.nonzero(inside, as_tuple=False).squeeze(1)
        if s2.numel() == 0:
            continue

        pix_l.append((py.long() * W + px.long()).reshape(-1).index_select(0, s2))
        a_l.append(w.reshape(-1).clamp(0, 0.999).index_select(0, s2))
        z_l.append(z[sl].unsqueeze(1).expand(-1, ox.shape[1]).reshape(-1).index_select(0, s2))
        c_l.append(
            col[sl].unsqueeze(1).expand(-1, ox.shape[1], -1).reshape(-1, 3).index_select(0, s2)
        )

    if not pix_l:
        return (
            torch.tensor(bg, device=dev, dtype=dt).view(1, 3).expand(H * W, 3).reshape(H, W, 3),
            torch.zeros(H, W, 1, device=dev, dtype=dt),
        )

    pix = torch.cat(pix_l)
    a = torch.cat(a_l)
    zz = torch.cat(z_l)
    cc = torch.cat(c_l)

    # Sortierung nach (Pixel, Tiefe): erst nach Tiefe, dann stabil nach Pixel
    by_z = torch.argsort(zz)
    pix, a, cc = pix.index_select(0, by_z), a.index_select(0, by_z), cc.index_select(0, by_z)
    by_pix = torch.argsort(pix, stable=True)
    pix, a, cc = pix.index_select(0, by_pix), a.index_select(0, by_pix), cc.index_select(0, by_pix)

    # Segment-Produkt der Transmission, im Lograum (stabiler als cumprod)
    one = torch.zeros(1, device=dev, dtype=dt)
    cl = torch.cat([one, torch.cumsum(torch.log((1.0 - a).clamp(min=1e-6)), 0)])
    starts = torch.searchsorted(pix, pix, right=False)

    idx = torch.arange(len(pix), device=dev)
    t_before = torch.exp(cl.index_select(0, idx) - cl.index_select(0, starts))
    contrib = cc * (a * t_before).unsqueeze(1)

    flat = torch.zeros(H * W, 3, device=dev, dtype=dt)
    img = flat.scatter_add(0, pix.unsqueeze(1).expand(-1, 3), contrib)

    # Deckung pro Pixel: 1 - prod(1-a) = 1 - exp(sum(log(1-a)))
    log_t = torch.zeros(H * W, 1, device=dev, dtype=dt)
    log_t = log_t.scatter_add(
        0, pix.unsqueeze(1), torch.log((1.0 - a).clamp(min=1e-6)).unsqueeze(1)
    )
    alpha_flat = 1.0 - torch.exp(log_t)

    bg_t = torch.tensor(bg, device=dev, dtype=dt).view(1, 3)
    img = img + bg_t * (1 - alpha_flat)
    return img.reshape(H, W, 3), alpha_flat.reshape(H, W, 1)
