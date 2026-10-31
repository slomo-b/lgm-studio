"""Kommandozeile: ein LGM-Splat rein, ein SimReady-USD raus.

    python -m simready build LGM/workspace_test/catstatue_rgba.ply \
        -o out/catstatue --class mug --target-size 0.09
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

from . import collider as collider_mod
from . import lint as lint_mod
from . import mesh as mesh_mod
from . import metric as metric_mod
from . import splat as splat_mod
from . import usd_export


def _safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_]", "_", s or "asset")
    return s if not s[0].isdigit() else "a_" + s


def cmd_build(args):
    t0 = time.time()
    name = _safe_name(args.name or os.path.splitext(os.path.basename(args.ply))[0])
    os.makedirs(args.out, exist_ok=True)
    steps = {}

    # 1) Splat -> Punktwolke
    g = splat_mod.load_gaussians(args.ply)
    g = splat_mod.clean(g, min_opacity=args.min_opacity, max_radius=args.max_radius)
    if args.voxel > 0:
        g = splat_mod.voxel_downsample(g, args.voxel)
    print(f"[1/5] Splat: {g['meta']['n_kept']:,}/{g['meta']['n_total']:,} Gaussians")
    steps["splat"] = g["meta"]

    # 2) Mesh
    if args.mesh_method == "tsdf":
        tm = mesh_mod.reconstruct_tsdf(
            g,
            views=args.views,
            res=args.tsdf_res,
            radius=args.cam_radius,
            target_faces=args.faces,
        )
    else:
        tm = mesh_mod.reconstruct(
            g,
            depth=args.poisson_depth,
            density_quantile=args.density_trim,
            k_normals=args.k_normals,
            target_faces=args.faces,
            normals=args.normals,
            orient=args.orient,
        )
        tm = mesh_mod.transfer_colors(tm, g)
    print(f"[2/5] Mesh: {len(tm.vertices):,} Vertices / {len(tm.faces):,} Faces")
    steps["mesh"] = mesh_mod.stats(tm)

    # 3) Metrik, Pivot, Z-up
    tm, minfo = metric_mod.normalize(
        tm,
        obj_class=args.obj_class,
        target_size=args.target_size,
        up_axis=args.up_axis,
        base=not args.no_base,
    )
    print(
        f"[3/5] Metrik: {minfo['target_size_m']:.3f} m "
        f"({minfo['size_source']}), extents = "
        f"{', '.join(f'{x*100:.1f}' for x in minfo['extents_m'])} cm"
    )
    steps["metric"] = minfo

    # 4) Collider (auf einer dezimierten Kopie -- CoACD skaliert mit Dreiecken)
    col_input = tm
    if args.collider_faces > 0 and len(tm.faces) > args.collider_faces:
        col_input = mesh_mod.decimate(tm, args.collider_faces)
    parts, mode = collider_mod.build(
        col_input,
        threshold=args.coacd_threshold,
        max_hulls=args.max_hulls,
        mode=args.collider,
    )
    cov = collider_mod.coverage(tm, parts)
    print(f"[4/5] Collider: {mode}, Abdeckung {cov:.2f}x")
    steps["collider"] = {
        "mode": mode,
        "n_parts": len(parts),
        "coverage": cov,
        "threshold": args.coacd_threshold,
    }

    # 5) USD + Lint
    usd_path = os.path.join(args.out, f"{name}.usda")
    usd_export.author(
        usd_path,
        name=name,
        visual=tm,
        colliders=parts,
        obj_class=args.obj_class or "generic",
        material_class=args.material_class,
        density=args.density,
        static_friction=args.static_friction,
        dynamic_friction=args.dynamic_friction,
        restitution=args.restitution,
        roughness=args.roughness,
        metallic=args.metallic,
        approx="convexHull",
        up_axis=args.up_axis,
    )
    print(f"[5/5] USD: {usd_path}")

    rep = lint_mod.check(usd_path)
    rep.print()

    # Mesh separat ablegen: praktisch fuer Sichtpruefung und Weiterverarbeitung
    obj_path = os.path.join(args.out, f"{name}.obj")
    tm.export(obj_path)

    if args.preview:
        _write_preview(tm, os.path.join(args.out, f"{name}_preview.png"))

    report = {
        "source_ply": os.path.abspath(args.ply),
        "asset": name,
        "obj_class": args.obj_class,
        "usd": os.path.abspath(usd_path),
        "mesh_obj": os.path.abspath(obj_path),
        "seconds": round(time.time() - t0, 2),
        "steps": steps,
        "lint": rep.to_dict(),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, type(None)))},
    }
    with open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFertig in {report['seconds']} s -> {args.out}")
    return 0 if rep.passed else 1


def _write_preview(tm, path, views=3, size=420):
    """Drei Ansichten als PNG, damit man das Mesh ohne Viewer beurteilen kann.

    Bewusst ein eigener 2D-Rasterizer statt Poly3DCollection: matplotlibs
    3D-Achsen sortieren Polygone nicht nach Tiefe, dadurch sieht jede
    geschlossene Oberflaeche wie eine durchsichtige Schale aus.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except Exception as e:
        print(f"[preview] matplotlib fehlt: {e}")
        return

    verts = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces)
    tris = verts[faces]

    # Flaechennormalen + Farbe pro Dreieck
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    cols = tm.visual.vertex_colors
    if cols is not None and len(cols) == len(verts):
        base = cols[faces].mean(axis=1)[:, :3] / 255.0
    else:
        base = np.tile([0.72, 0.72, 0.75], (len(faces), 1))

    center = verts.mean(axis=0)
    radius = float(np.max(np.linalg.norm(verts - center, axis=1))) * 2.6
    light = np.array([0.45, 0.65, 0.62])
    light /= np.linalg.norm(light)

    azims = [30, 120, 210][:views]
    fig, axes = plt.subplots(1, views, figsize=(3.6 * views, 3.8), dpi=110)
    axes = np.atleast_1d(axes)

    for ax, az in zip(axes, azims):
        a, e = np.deg2rad(az), np.deg2rad(18.0)
        # Z-up: die Metrikstufe hat das Mesh bereits gedreht, der Orbit muss
        # also um Z laufen, nicht um Y.
        eye = center + radius * np.array(
            [np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)]
        )
        fwd = center - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)

        d = tris - eye
        z = d @ fwd
        u = (d @ right) / np.maximum(z, 1e-6)
        v = (d @ up) / np.maximum(z, 1e-6)

        # Bildweite aus der tatsaechlichen Ausdehnung, nicht aus dem Kameraradius
        dv = verts - eye
        zv = np.maximum(dv @ fwd, 1e-6)
        uv = (dv @ right) / zv
        vv = (dv @ up) / zv
        lim = float(np.percentile(np.abs(np.concatenate([uv, vv])), 99.5)) * 1.12
        lim = max(lim, 1e-3)

        # Kein Backface-Culling: die Winding-Reihenfolge ist bei Poisson-Meshes
        # nicht verlaesslich, Culling wuerde Loecher vortaeuschen. Die
        # Tiefensortierung entscheidet stattdessen, was vorne liegt.
        facing = np.ones(len(faces), dtype=bool)
        zmin = z.mean(axis=1)
        order = np.argsort(-zmin)  # hinten zuerst
        keep = order[facing[order]]

        polys = np.stack([u[keep], v[keep]], axis=2)
        shade = 0.35 + 0.65 * np.clip(np.abs(n[keep] @ light), 0, 1)
        fc = np.clip(base[keep] * shade[:, None], 0, 1)

        ax.add_collection(PolyCollection(polys, facecolors=fc, edgecolors="none", antialiased=False))
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(f"az {az}°", fontsize=9)

    fig.tight_layout(pad=0.2)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[preview] {path}")


def build_parser():
    p = argparse.ArgumentParser(prog="simready", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Splat -> SimReady-USD")
    b.add_argument("ply", help="3DGS-.ply von LGM")
    b.add_argument("-o", "--out", required=True, help="Zielordner")
    b.add_argument("--name", default=None, help="Asset-Name / Prim-Name")

    b.add_argument("--obj-class", default=None, help="Klasse fuer Groessen-Prior (mug, chair, ...)")
    b.add_argument("--target-size", type=float, default=None, help="laengste Kante in Metern")
    b.add_argument("--material-class", default="generic", help="wood, metal, plastic, ...")
    b.add_argument("--density", type=float, default=500.0, help="kg/m^3 fuer die Masse")

    b.add_argument("--min-opacity", type=float, default=0.05)
    b.add_argument("--max-radius", type=float, default=1.15)
    b.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Voxel-Downsampling der Punktwolke; 0 = aus. Bei LGM sind ~0.008 sinnvoll",
    )
    b.add_argument(
        "--mesh-method",
        default="tsdf",
        choices=["tsdf", "poisson"],
        help="tsdf = Tiefenkarten fusionieren (geschlossen), poisson = Punktwolke",
    )
    b.add_argument("--views", type=int, default=36, help="Ansichten fuer TSDF")
    b.add_argument("--tsdf-res", type=int, default=160, help="Voxelraster pro Achse")
    b.add_argument("--cam-radius", type=float, default=2.4)
    b.add_argument("--poisson-depth", type=int, default=8)
    b.add_argument("--density-trim", type=float, default=0.02)
    b.add_argument("--k-normals", type=int, default=30)
    b.add_argument(
        "--normals",
        default="knn",
        choices=["knn", "covariance"],
        help="Normalenschaetzung; 'covariance' nutzt die Gaussian-Kovarianz",
    )
    b.add_argument(
        "--orient",
        default="perpoint",
        choices=["perpoint", "consensus", "none"],
        help="Orientierung der Normalen vor Poisson",
    )
    b.add_argument("--faces", type=int, default=60000, help="0 = nicht dezimieren (Anzeige-Mesh)")
    b.add_argument(
        "--collider-faces",
        type=int,
        default=8000,
        help="Dreiecke fuer CoACD; 0 = volles Mesh (langsam)",
    )

    b.add_argument("--up-axis", default="Z", choices=["Z", "Y"])
    b.add_argument("--no-base", action="store_true", help="Pivot nicht auf die Unterkante setzen")

    b.add_argument("--collider", default="auto", choices=["auto", "coacd", "hull", "none"])
    b.add_argument("--coacd-threshold", type=float, default=0.05)
    b.add_argument("--max-hulls", type=int, default=32)

    b.add_argument("--static-friction", type=float, default=0.8)
    b.add_argument("--dynamic-friction", type=float, default=0.6)
    b.add_argument("--restitution", type=float, default=0.1)
    b.add_argument("--roughness", type=float, default=0.6)
    b.add_argument("--metallic", type=float, default=0.0)
    b.add_argument("--preview", action="store_true", help="PNG-Vorschau schreiben")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("lint", help="bestehendes USD pruefen")
    c.add_argument("usd")
    c.set_defaults(func=lambda a: (lint_mod.check(a.usd).print(), 0)[1])
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
