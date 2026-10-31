"""Kollisionskoerper: konvexe Zerlegung.

PhysX kollidiert nicht mit Dreieckssuppen, sondern mit konvexen Koerpern. Ein
einzelner Convex-Hull eines Stuhls ist ein Quader -- CoACD zerlegt die
Oberflaeche stattdessen in mehrere konvexe Stuecke und trifft die Form deutlich
besser.
"""

import numpy as np
import trimesh


def convex_decomposition(
    tm,
    threshold=0.05,
    max_hulls=32,
    seed=0,
    prep_resolution=30,
    mcts_iterations=100,
    max_ch_vertex=64,
    resolution=150,
    mcts_nodes=8,
    mcts_max_depth=2,
):
    """CoACD. Rueckgabe: Liste von trimesh.Trimesh (jeweils konvex).

    Laufzeit haengt fast linear an der Dreieckszahl des Inputs -- deshalb
    bekommt CoACD in der CLI eine separat dezimierte Kopie, nicht das
    Anzeige-Mesh. `resolution` ist die Stichprobenaufloesung der MCTS-Suche und
    mit Abstand der teuerste Parameter (Default 2000 ist fuer Assets zu viel).
    """
    import coacd

    m = coacd.Mesh(
        np.asarray(tm.vertices, dtype=np.float64),
        np.asarray(tm.faces, dtype=np.int32),
    )
    parts = coacd.run_coacd(
        m,
        threshold=float(threshold),
        max_convex_hull=int(max_hulls),
        preprocess_resolution=int(prep_resolution),
        resolution=int(resolution),
        mcts_nodes=int(mcts_nodes),
        mcts_iterations=int(mcts_iterations),
        mcts_max_depth=int(mcts_max_depth),
        max_ch_vertex=int(max_ch_vertex),
        decimate=True,
        seed=int(seed),
        merge=True,
    )
    out = []
    for verts, faces in parts:
        part = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        if len(part.faces) >= 4:
            out.append(part)
    return out


def hull(tm):
    """Fallback: ein einzelner konvexer Hull."""
    h = tm.convex_hull
    return [trimesh.Trimesh(vertices=h.vertices, faces=h.faces, process=False)]


def build(
    tm,
    threshold=0.05,
    max_hulls=32,
    mode="auto",
    prep_resolution=30,
    mcts_iterations=100,
    convex_tol=1.25,
):
    """mode: 'auto' | 'coacd' | 'hull' | 'none'.

    'auto' ist der Default, weil die meisten LGM-Ausgaben blob-foermig und
    nahezu konvex sind: dann ist ein einzelner Hull in Millisekunden fertig und
    physikalisch ausreichend. Nur bei echt konkaven Formen lohnt CoACD.
    """
    if mode == "none":
        return [], "none"
    if mode == "hull":
        return hull(tm), "convex hull (erzwungen)"

    c = convexity(tm)
    if mode == "auto" and c is not None and c < convex_tol:
        return hull(tm), f"convex hull (Konvexitaet {c:.2f} < {convex_tol})"
    try:
        parts = convex_decomposition(
            tm,
            threshold=threshold,
            max_hulls=max_hulls,
            prep_resolution=prep_resolution,
            mcts_iterations=mcts_iterations,
        )
        if not parts:
            return hull(tm), "convex hull (CoACD leer)"
        reason = f"coacd ({len(parts)} Teile"
        reason += f", Konvexitaet {c:.2f})" if c is not None else ")"
        return parts, reason
    except Exception as e:
        print(f"[collider] CoACD fehlgeschlagen ({e}) -> Convex Hull")
        return hull(tm), "convex hull (CoACD Fehler)"


def convexity(tm):
    """Hull-Volumen / Mesh-Volumen. 1.0 = konvex, >1.25 = deutlich konkav.

    Bewusst nicht an `is_volume` gekoppelt: die Divergenz-Formel liefert auch
    fuer eine nicht ganz geschlossene Oberflaeche eine brauchbare Abschaetzung,
    und genau die brauchen wir fuer die Entscheidung Hull vs. CoACD.
    """
    try:
        hv = float(abs(tm.convex_hull.volume))
        v = float(abs(tm.volume))
    except Exception:
        return None
    if not np.isfinite(v) or not np.isfinite(hv) or v <= 1e-12 or hv <= 1e-12:
        return None
    return hv / v


def coverage(tm, parts):
    """Wie viel der sichtbaren Oberflaeche die Collider abdecken (grob)."""
    if not parts:
        return 0.0
    v_coll = sum(float(p.volume) for p in parts if p.is_volume)
    try:
        v_mesh = float(tm.volume) if tm.is_volume else float(tm.convex_hull.volume)
    except Exception:
        return 0.0
    if v_mesh <= 1e-12:
        return 0.0
    return float(np.clip(v_coll / v_mesh, 0.0, 3.0))
