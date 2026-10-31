"""TSDF-Rekonstruktion aus gerenderten Tiefenkarten.

Warum nicht Poisson: Poisson braucht verlaessliche Normalen, und die sind aus
einem 3DGS nicht zu gewinnen -- LGM konkateniert vier Ansichts-Schalen, deren
Oberflaechen sich widersprechen. TSDF braucht **keine** Normalen. Es rendert
Tiefenkarten, projiziert sie zurueck und mittelt die Messungen in einem
Voxelgitter. Das Ergebnis ist per Konstruktion geschlossen.

Nebeneffekt: dieselben Tiefenkarten sind die Supervision fuer den
Depth/Normal-Head in Stufe B (siehe ANALYSE_SIMREADY.md, B3).

Alles hier ist reines numpy -- kein torch, keine GPU.
"""

import numpy as np

FOVY = 49.1  # identisch zu LGM (core/options.py)


def orbit_cameras(n_views=36, elev_range=(2.0, 32.0), radius=2.4, center=None):
    """Kameras auf einer Kugelschale, Y-up (LGM-Welt, vor der Metrikstufe).

    Rueckgabe: Liste von (R_cv, eye); R_cv ist die Rotation Welt->Kamera in
    OpenCV-Konvention (x rechts, y unten, z nach vorn).
    """
    center = np.zeros(3) if center is None else np.asarray(center, dtype=np.float64)
    cams = []
    for i in range(int(n_views)):
        # goldener Winkel -> gleichmaessige Verteilung ohne Reihen-Artefakte
        a = np.deg2rad((i * 137.508) % 360.0)
        e = np.deg2rad(elev_range[0] + (elev_range[1] - elev_range[0]) * ((i * 0.618) % 1.0))
        eye = center + radius * np.array(
            [np.cos(e) * np.sin(a), np.sin(e), np.cos(e) * np.cos(a)]
        )
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
        n = np.linalg.norm(right)
        right = right / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        down = np.cross(fwd, right)
        cams.append((np.stack([right, down, fwd], axis=0), eye))
    return cams


def intrinsic(size, fovy=FOVY):
    f = (size / 2.0) / np.tan(0.5 * np.deg2rad(fovy))
    return f, size / 2.0, size / 2.0


def render_depth(
    xyz, color, size=256, fovy=FOVY, R=None, eye=None, near=0.05, smooth=5,
    footprint=2,
):
    """Z-Buffer: pro Pixel die vorderste Gaussian.

    `footprint` rastert jeden Punkt als Scheibe statt als Einzelpixel. Der
    Punktabstand liegt bei ~1,4 px; ohne Scheibe bleiben Loecher, durch die der
    Z-Buffer die *Rueckseite* des Objekts sieht -- und ein TSDF carvt dann quer
    durchs Innere.

    `smooth` medianfiltert anschliessend gegen Ausreisser an Kanten.

    Rueckgabe: depth [H,W] (Kamerazaehler, 0 = leer), rgb [H,W,3] uint8.
    """
    f, cx, cy = intrinsic(size, fovy)
    d = xyz - eye
    pc = d @ R.T  # [N,3]
    z = pc[:, 2]
    ok = z > near

    u = np.zeros(len(z), dtype=np.int64)
    v = np.zeros(len(z), dtype=np.int64)
    u[ok] = np.round(f * pc[ok, 0] / z[ok] + cx).astype(np.int64)
    v[ok] = np.round(f * pc[ok, 1] / z[ok] + cy).astype(np.int64)
    ok &= (u >= -int(footprint)) & (u < size + int(footprint))
    ok &= (v >= -int(footprint)) & (v < size + int(footprint))

    if not np.any(ok):
        return np.zeros((size, size), np.float32), np.zeros((size, size, 3), np.uint8)

    # Offsets fuer die Punktscheibe, einmal vorberechnet
    fp = int(footprint)
    offs = [
        (dy, dx)
        for dy in range(-fp, fp + 1)
        for dx in range(-fp, fp + 1)
        if dy * dy + dx * dx <= fp * fp
    ]
    uu = u[ok][None, :] + np.array([o[1] for o in offs])[:, None]
    vv = v[ok][None, :] + np.array([o[0] for o in offs])[:, None]
    zz = np.broadcast_to(z[ok][None, :], uu.shape)

    inside = (uu >= 0) & (uu < size) & (vv >= 0) & (vv < size)
    pix = (vv * size + uu)[inside]
    zf = zz[inside].astype(np.float32)
    cs = color[ok].astype(np.float32)
    cf = np.broadcast_to(cs[None, :, :], (len(offs), cs.shape[0], 3))[inside]

    flat = np.zeros(size * size, dtype=np.float32)
    rgb = np.zeros((size * size, 3), dtype=np.float32)
    if pix.size == 0:
        return flat.reshape(size, size), np.zeros((size, size, 3), np.uint8)

    order = np.argsort(zf, kind="stable")
    pix, zf, cf = pix[order], zf[order], cf[order]
    uniq, first = np.unique(pix, return_index=True)
    flat[uniq] = zf[first]
    rgb[uniq] = cf[first]

    # kleine Luecken schliessen: pro Loch den naechsten Nachbarn nehmen
    from scipy import ndimage

    depth = flat.reshape(size, size)
    color_img = rgb.reshape(size, size, 3)
    for _ in range(2):
        valid = depth > 0
        if valid.all():
            break
        big = np.where(valid, depth, 1e6).astype(np.float32)
        near_d = ndimage.grey_erosion(big, size=3)
        fill = (~valid) & (near_d < 1e5)
        if not fill.any():
            break
        depth = np.where(fill, near_d, depth)
        # Farbe aus der Nachbarschaft uebernehmen
        for c in range(3):
            ch = np.where(valid, color_img[..., c], -1.0).astype(np.float32)
            mx = ndimage.maximum_filter(ch, size=3)
            color_img[..., c] = np.where(fill & (mx >= 0), mx, color_img[..., c])

    if smooth and int(smooth) > 1:
        k = int(smooth)
        valid = depth > 0
        sm = ndimage.median_filter(
            np.where(valid, depth, 0.0).astype(np.float32), size=k, mode="nearest"
        )
        cov = ndimage.uniform_filter(valid.astype(np.float32), size=k, mode="nearest")
        depth = np.where(valid & (cov > 0.3) & (sm > 0), sm, 0.0).astype(np.float32)

    return depth, np.clip(color_img * 255.0, 0, 255).astype(np.uint8)


def tsdf_fuse(
    xyz,
    color,
    n_views=36,
    size=256,
    radius=2.4,
    res=160,
    sdf_trunc_vox=4.0,
    pad=0.12,
    fovy=FOVY,
    verbose=True,
):
    """Eigener TSDF: Tiefenkarten projizieren, SDF mitteln, Marching Cubes.

    Open3Ds `ScalableTSDFVolume`/`UniformTSDFVolume` liefern in dieser
    Installation null Faces (getestet mit synthetischer Kugel und korrekten
    Extrinsics), deshalb hier die eigene, kurze Variante. Sie ist auch
    nachvollziehbar: pro Voxel wird die Differenz zwischen beobachteter Tiefe
    und Voxeltiefe geclippt und gemittelt.
    """
    import numpy as np
    from skimage import measure

    xyz = np.asarray(xyz, dtype=np.float64)
    lo = xyz.min(axis=0) - pad
    hi = xyz.max(axis=0) + pad
    extent = hi - lo
    vox = float(extent.max()) / float(res)
    dims = np.maximum(np.ceil(extent / vox).astype(int), 4)
    sdf_trunc = max(float(sdf_trunc_vox) * vox, 1e-6)

    axes = [lo[i] + (np.arange(dims[i]) + 0.5) * vox for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n_vox = grid.shape[0]

    f, cx, cy = intrinsic(size, fovy)
    center = xyz.mean(axis=0)
    cams = orbit_cameras(n_views, radius=radius, center=center)

    occ = np.ones(n_vox, dtype=bool)

    chunk = 1_000_000
    for vi, (R, eye) in enumerate(cams):
        depth, _ = render_depth(xyz, color, size=size, fovy=fovy, R=R, eye=eye)
        if not np.any(depth > 0):
            continue

        # Die Tiefenkarte hat Loecher *innerhalb* der Silhouette (nicht jedes
        # Pixel bekommt eine Gaussian). Ungefuellt wuerde jedes Loch einen
        # Tunnel durch das Voxelvolumen stanzen -- das Volumen schrumpft dann
        # auf eine Schale. Innerhalb der Silhouette also mit dem naechsten
        # gueltigen Tiefenwert auffuellen.
        from scipy import ndimage as _nd

        # Die Tiefenkarte hat Loecher *innerhalb* der Silhouette (nicht jedes
        # Pixel bekommt eine Gaussian). Dort sieht der Z-Buffer die RUECKSEITE
        # des Objekts (1,9 statt 1,1) -- solche Ausreisser carven dann quer
        # durchs Innere. Erst fliegende Pixel verwerfen, dann fuellen.
        from scipy import ndimage as _nd

        front = depth > 0
        if front.any():
            med_val = float(np.median(depth[front]))
            probe = np.where(front, depth, med_val)
            med = _nd.median_filter(probe, size=5)
            flying = front & (np.abs(depth - med) > 3.0 * sdf_trunc)
            depth[flying] = 0.0

        valid = depth > 0
        # Der Medianfilter in render_depth erodiert die Maske am Rand, dadurch
        # haengen Loecher mit dem Hintergrund zusammen und binary_fill_holes
        # greift nicht. Erst schliessen, dann fuellen.
        sil = _nd.binary_fill_holes(_nd.binary_dilation(valid, iterations=2))
        near_idx = _nd.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        depth = np.where(sil, depth[tuple(near_idx)], 0.0).astype(np.float32)

        for start in range(0, n_vox, chunk):
            blk = grid[start : start + chunk]
            pc = (blk - eye) @ R.T
            z = pc[:, 2]
            ok = z > 1e-3
            u = np.full(len(z), -1, dtype=np.int64)
            v = np.full(len(z), -1, dtype=np.int64)
            u[ok] = np.round(f * pc[ok, 0] / z[ok] + cx).astype(np.int64)
            v[ok] = np.round(f * pc[ok, 1] / z[ok] + cy).astype(np.int64)
            ok &= (u >= 0) & (u < size) & (v >= 0) & (v < size)
            if not ok.any():
                continue
            obs = np.zeros(len(z), dtype=np.float32)
            obs[ok] = depth[v[ok], u[ok]]

            # 1) Carving: ausserhalb der Silhouette = definitiv leer.
            occ[start + np.flatnonzero(ok & (obs <= 0))] = False

            # 2) Carving: deutlich vor der beobachteten Oberflaeche = leer.
            seen = ok & (obs > 0)
            if seen.any():
                s = obs[seen] - z[seen]
                si = np.flatnonzero(seen)
                occ[start + si[s > sdf_trunc]] = False
        if verbose and (vi + 1) % 12 == 0:
            print(f"[tsdf] {vi + 1}/{len(cams)} Ansichten fusioniert")

    if not occ.any():
        raise RuntimeError("Carving hat alles entfernt -- Silhouetten passen nicht")

    # Belegtheit als 0/1-Volumen, dann glaetten: die Nullflaeche bei 0.5 wird
    # dadurch stetig und Marching Cubes liefert keine Treppen.
    from scipy import ndimage

    vol = ndimage.gaussian_filter(
        occ.astype(np.float32).reshape(dims), sigma=1.2, mode="nearest"
    )

    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=(vox, vox, vox))
    verts = verts + lo
    if verbose:
        print(
            f"[tsdf] {float(occ.mean()):.1%} der Voxel belegt, "
            f"{len(faces):,} Faces bei vox={vox:.4f}"
        )

    import trimesh

    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)

