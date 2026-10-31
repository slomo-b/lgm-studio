"""3DGS-.ply laden und in eine Punktwolke mit Normalen umrechnen.

Der Trick: eine Gaussian hat bereits eine 3x3-Kovarianz
`C = R diag(s^2) R^T`. Ihr Eigenvektor zum *kleinsten* Eigenwert ist die
Oberflaechennormale -- die Gaussian ist in dieser Richtung am flachsten. Damit
brauchen wir weder Tiefenrenderings noch eine Normalschaetzung aus Nachbarn.
"""

import numpy as np

# SH-Konstante: f_dc -> RGB (Grad 0), wie in der 3DGS-Referenz
SH_C0 = 0.28209479177387814


def quat_to_rotmat(q):
    """[N,4] (w,x,y,z) -> [N,3,3]."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def load_gaussians(path):
    """Liest ein 3DGS-.ply (Standardformat) und gibt aktivierte Werte zurueck."""
    from plyfile import PlyData

    field = PlyData.read(path).elements[0].data
    names = set(field.dtype.names or ())

    xyz = np.stack([field["x"], field["y"], field["z"]], axis=1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(field["opacity"], dtype=np.float32)))
    scale = np.exp(
        np.stack([field["scale_0"], field["scale_1"], field["scale_2"]], axis=1).astype(np.float32)
    )
    # LGM schreibt bereits normalisierte Quaternionen, aber wir normalisieren defensiv
    rot = np.stack(
        [field["rot_0"], field["rot_1"], field["rot_2"], field["rot_3"]], axis=1
    ).astype(np.float32)

    # LGM nutzt SH-Grad 0; hoehere Baender werden ignoriert (siehe ANALYSE_SIMREADY.md B2)
    f_dc = np.stack([field["f_dc_0"], field["f_dc_1"], field["f_dc_2"]], axis=1).astype(np.float32)
    color = np.clip(SH_C0 * f_dc + 0.5, 0.0, 1.0)

    meta = {"has_sh_rest": any(n.startswith("f_rest_") for n in names)}
    return {"xyz": xyz, "opacity": opacity, "scale": scale, "rot": rot, "color": color, "meta": meta}


def normals_from_covariance(rot, scale):
    """Kleinster Eigenvektor der 3D-Kovarianz, pro Gaussian."""
    R = quat_to_rotmat(rot)
    # C = R @ diag(s^2) @ R^T  ->  Spalten von R mit s^2 skalieren
    M = R * (scale * scale)[:, None, :]
    cov = M @ R.transpose(0, 2, 1)
    # eigh sortiert aufsteigend -> [:, :, 0] ist die Normale
    _, vecs = np.linalg.eigh(cov.astype(np.float64))
    return vecs[:, :, 0].astype(np.float32)


def normals_knn(points, k=30):
    """Normalen aus der lokalen Nachbarschaft (PCA der k naechsten Punkte).

    Robuster als die Kovarianz: LGM-Gaussians sind oft fast isotrop, dann zeigt
    ihre kleinste Achse in eine beliebige Richtung und Poisson produziert Muell.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.0, max_nn=int(k))
    )
    return np.asarray(pcd.normals).astype(np.float32)


def estimate_normals(g, method="knn", k=30):
    if method == "covariance":
        return normals_from_covariance(g["rot"], g["scale"])
    return normals_knn(g["xyz"], k=k)


def clean(g, min_opacity=0.05, max_radius=1.15, min_scale=1e-5):
    """Offensichtlichen Muell entfernen: fast transparente und entartete Gaussians."""
    keep = (g["opacity"] >= min_opacity) & (np.linalg.norm(g["xyz"], axis=1) <= max_radius)
    keep &= np.min(g["scale"], axis=1) >= min_scale
    n_before = g["xyz"].shape[0]
    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[0] == n_before else v) for k, v in g.items()}
    out["meta"] = dict(g["meta"], n_total=n_before, n_kept=int(keep.sum()))
    return out


def voxel_downsample(g, voxel):
    """Punktwolke ausduennen und dabei Duplikate zusammenfassen.

    LGM konkateniert die Vorhersagen von vier Ansichten. An jeder Stelle der
    Oberflaeche liegen dadurch bis zu vier fast identische Punkte -- und eine
    lokale PCA ueber solche Duplikate liefert zufaellige Normalen.
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(g["xyz"], dtype=np.float64))
    if "color" in g:
        pcd.colors = o3d.utility.Vector3dVector(
            np.asarray(g["color"], dtype=np.float64)
        )
    out = pcd.voxel_down_sample(float(voxel))
    pts = np.asarray(out.points).astype(np.float32)
    cols = np.asarray(out.colors).astype(np.float32)
    if cols.size == 0:
        cols = np.full((len(pts), 3), 0.7, dtype=np.float32)

    n = len(pts)
    return {
        "xyz": pts,
        "color": np.clip(cols, 0, 1),
        # nach dem Downsampling werden diese Felder nicht mehr gebraucht
        "opacity": np.ones((n, 1), dtype=np.float32),
        "scale": np.full((n, 3), float(voxel), dtype=np.float32),
        "rot": np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (n, 1)),
        "meta": dict(g.get("meta", {}), downsampled=True, n_kept=n, voxel=float(voxel)),
    }


def to_open3d(points, normals, colors=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals, dtype=np.float64))
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return pcd


def orient_outward(points, normals):
    """Globales Vorzeichen der Normalen bestimmen.

    `orient_normals_consistent_tangent_plane` macht die Normalen untereinander
    konsistent, laesst aber das globale Vorzeichen offen. Wir drehen sie so,
    dass sie im Mittel vom Schwerpunkt wegzeigen -- richtig fuer alles, was
    ueberwiegend sternfoermig ist, falsch fuer tiefe Hohlraeume (Tasse innen).
    """
    c = points.mean(axis=0)
    vote = np.einsum("ij,ij->i", normals, points - c).sum()
    return -normals if vote < 0 else normals


def orient_outward_per_point(points, normals):
    """Jede Normale einzeln vom Schwerpunkt weg drehen.

    Noetig, weil LGM vier Ansichts-Schalen liefert, die als getrennte
    Komponenten keine gemeinsame Orientierung bekommen: die eine zeigt nach
    innen, die andere nach aussen, und Poisson macht daraus Doppelflaechen.
    """
    c = points.mean(axis=0)
    s = np.einsum("ij,ij->i", normals, points - c)
    out = np.array(normals, copy=True)
    out[s < 0] *= -1.0
    return out
