"""Poisson-Rekonstruktion aus der Splat-Punktwolke, plus Reparatur.

Poisson ist hier die richtige Wahl: es erzeugt per Konstruktion eine
*geschlossene* Oberflaeche, und genau das verlangt ein Kollisionskoerper.
Der Preis sind aufgeblaehte Details und -- bei verrauschten Punkten --
nicht-mannigfaltige Raender. Deshalb: Density-Trim, Aufraeumen in Open3D
(dort gibt es `remove_non_manifold_edges`) und erst danach Decimation.
"""

import numpy as np
import trimesh

from . import splat as splat_mod


def to_trimesh(o3d_mesh):
    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def to_o3d(tm):
    import open3d as o3d

    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32))
    return m


def _clean_o3d(mesh, manifold_fix=True):
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    if manifold_fix:
        # Achtung: oeffnet die Oberflaeche, wenn es nicht-mannigfaltige Kanten
        # gibt. Bei TSDF-Meshes deshalb abschaltbar.
        mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    return mesh


def _largest_component_o3d(mesh):
    """Groesste zusammenhaengende Insel behalten -- ohne networkx-Umweg."""
    labels, _, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    areas = np.asarray(areas)
    if labels.size == 0 or areas.size <= 1:
        return mesh
    keep = int(np.argmax(areas))
    mesh.remove_triangles_by_mask(labels != keep)
    mesh.remove_unreferenced_vertices()
    return mesh


def reconstruct(
    g,
    depth=8,
    density_quantile=0.02,
    k_normals=30,
    target_faces=0,
    normals="knn",
    orient="perpoint",
):
    """g: Ausgabe von splat.clean(). Rueckgabe: trimesh.Trimesh in [-1,1]^3."""
    import open3d as o3d

    points = np.asarray(g["xyz"], dtype=np.float64)
    nrm = splat_mod.estimate_normals(g, method=normals, k=k_normals)

    pcd = splat_mod.to_open3d(points, nrm, g["color"])
    if orient == "consensus" and k_normals > 0:
        # global konsistente Orientierung; ohne das liefert Poisson Muell
        pcd.orient_normals_consistent_tangent_plane(int(k_normals))
        pcd.normals = o3d.utility.Vector3dVector(
            splat_mod.orient_outward(points, np.asarray(pcd.normals))
        )
    elif orient == "perpoint":
        pcd.normals = o3d.utility.Vector3dVector(
            splat_mod.orient_outward_per_point(points, nrm)
        )

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=int(depth), scale=1.1, linear_fit=False
    )
    n_raw = len(mesh.triangles)

    # Poisson erfindet Oberflaeche dort, wo keine Punkte sind: niedrige Density
    # = Extrapolation -> wegschneiden.
    d = np.asarray(densities)
    if density_quantile > 0 and d.size:
        mesh.remove_vertices_by_mask(d < np.quantile(d, float(density_quantile)))

    mesh = _clean_o3d(mesh)
    mesh = _largest_component_o3d(mesh)
    print(f"[mesh] Poisson depth={depth}: {n_raw:,} -> {len(mesh.triangles):,} Faces")

    if target_faces and len(mesh.triangles) > target_faces:
        mesh = decimate_o3d(mesh, target_faces)

    tm = repair(to_trimesh(mesh))
    if len(tm.faces) == 0:
        raise RuntimeError("Nach der Reparatur blieb kein Mesh uebrig")
    return tm


def reconstruct_tsdf(
    g, views=36, size=256, res=160, radius=2.4, target_faces=0, verbose=True
):
    """TSDF-Weg: ohne Normalen, per Konstruktion geschlossen."""
    from . import depth

    tm = depth.tsdf_fuse(
        np.asarray(g["xyz"], dtype=np.float64),
        np.asarray(g["color"], dtype=np.float32),
        n_views=int(views),
        size=int(size),
        radius=float(radius),
        res=int(res),
        verbose=verbose,
    )
    tm = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=True)
    tm.merge_vertices()
    tm.update_faces(tm.unique_faces())
    tm = repair(tm)
    if target_faces and len(tm.faces) > target_faces:
        tm = decimate(tm, target_faces)
    # Farben erst NACH der Dezimierung: Open3Ds Decimator verwirft Vertexfarben.
    tm = transfer_colors(tm, g)
    if len(tm.faces) == 0:
        raise RuntimeError("TSDF lieferte kein Mesh")
    return tm


def decimate_o3d(mesh, target_faces):

    before = len(mesh.triangles)
    out = mesh.simplify_quadric_decimation(int(target_faces))
    out = _clean_o3d(out)
    print(f"[mesh] decimated {before:,} -> {len(out.triangles):,} Faces")
    return out


def decimate(tm, target_faces):
    if len(tm.faces) <= target_faces:
        return tm
    return to_trimesh(decimate_o3d(to_o3d(tm), target_faces))


def repair(tm, fill=True, smooth=0):
    """Normale ausrichten, Loecher schliessen, optional glaetten."""
    tm = trimesh.Trimesh(vertices=tm.vertices, faces=tm.faces, process=True)
    tm.merge_vertices()
    tm.update_faces(tm.nondegenerate_faces())
    tm.update_faces(tm.unique_faces())
    tm.remove_unreferenced_vertices()

    if fill:
        # fill_holes braucht eine weitgehend geschlossene Oberflaeche;
        # bei offenen Raendern einfach ignorieren statt abzubrechen.
        try:
            trimesh.repair.fill_holes(tm)
        except Exception as e:
            print(f"[mesh] fill_holes uebersprungen: {e}")

    trimesh.repair.fix_normals(tm)

    if smooth > 0:
        try:
            trimesh.smoothing.filter_taubin(tm, lamb=0.5, nu=-0.53, iterations=int(smooth))
        except Exception as e:
            print(f"[mesh] Glaettung uebersprungen: {e}")

    return tm


def transfer_colors(tm, g, knn=4):
    """Vertexfarben von den naechsten Splats uebernehmen (Poisson liefert keine)."""
    from scipy.spatial import cKDTree

    tree = cKDTree(g["xyz"])
    k = min(int(knn), g["xyz"].shape[0])
    _, idx = tree.query(np.asarray(tm.vertices), k=k)
    idx = np.atleast_2d(idx)
    cols = g["color"][idx].mean(axis=1)
    tm.visual = trimesh.visual.ColorVisuals(mesh=tm, vertex_colors=np.clip(cols, 0, 1))
    return tm


def stats(tm):
    def safe(fn, default=False):
        try:
            return bool(fn())
        except Exception:
            return default

    return {
        "vertices": int(len(tm.vertices)),
        "faces": int(len(tm.faces)),
        "watertight": safe(lambda: tm.is_watertight),
        "winding_consistent": safe(lambda: tm.is_winding_consistent),
        "volume": float(abs(tm.volume)) if safe(lambda: tm.is_volume) else None,
        "extents": [float(x) for x in tm.extents],
    }
