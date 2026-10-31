"""Lokaler Spiegel der pruefbaren Teilmenge der SimReady-Regeln.

WICHTIG: Das ist *nicht* der offizielle Validator. NVIDIA veroeffentlicht die
Regeln maschinenpruefbar in `NVIDIA/simready-foundation`; ein vollstaendiger
Konformitaetslauf gehoert dorthin. Hier werden nur die Regeln geprueft, die
sich ohne Simulator und ohne das Profil-Paket entscheiden lassen -- dafuer
sofort, offline und in der Pipeline.

Regel-IDs folgen der Nummerierung der Spezifikation, damit ein Report hier und
ein Report dort vergleichbar bleiben.
"""

import math

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


class Report:
    def __init__(self):
        self.rows = []

    def add(self, rule, ok, detail=""):
        self.rows.append({"rule": rule, "ok": bool(ok), "detail": detail})

    @property
    def passed(self):
        return all(r["ok"] for r in self.rows)

    def to_dict(self):
        return {
            "passed": self.passed,
            "checks": self.rows,
            "n_pass": sum(1 for r in self.rows if r["ok"]),
            "n_fail": sum(1 for r in self.rows if not r["ok"]),
        }

    def print(self):
        print("\n== SimReady-Lint (lokaler Spiegel) ==")
        for r in self.rows:
            mark = "PASS" if r["ok"] else "FAIL"
            line = f"  [{mark}] {r['rule']:<12} {r['detail']}"
            print(line)
        print(f"  -> {self.to_dict()['n_pass']} ok, {self.to_dict()['n_fail']} offen")


def check(stage_path):
    stage = Usd.Stage.Open(str(stage_path))
    rep = Report()
    if stage is None:
        rep.add("IO.000", False, f"Stage nicht lesbar: {stage_path}")
        return rep

    # --- Core / Units ------------------------------------------------------
    up = UsdGeom.GetStageUpAxis(stage)
    rep.add("UN.006", up == UsdGeom.Tokens.z, f"upAxis = {up} (erwartet Z)")

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    rep.add("UN.001", mpu is not None and abs(mpu - 1.0) < 1e-9, f"metersPerUnit = {mpu}")

    try:
        kg = UsdPhysics.GetStageKilogramsPerUnit(stage)
        rep.add("UN.002", kg is not None and abs(kg - 1.0) < 1e-9, f"kilogramsPerUnit = {kg}")
    except Exception as e:
        rep.add("UN.002", False, f"kilogramsPerUnit nicht lesbar: {e}")

    # --- Hierarchie --------------------------------------------------------
    default_prim = stage.GetDefaultPrim()
    rep.add(
        "ST.001",
        bool(default_prim and default_prim.IsValid()),
        f"defaultPrim = {default_prim.GetPath() if default_prim else 'fehlt'}",
    )

    # --- Visualisierung / Geometrie ---------------------------------------
    meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    rep.add("VG.001", len(meshes) > 0, f"{len(meshes)} Mesh-Prims")

    missing_normals = []
    for p in meshes:
        m = UsdGeom.Mesh(p)
        n = m.GetNormalsAttr().Get()
        if n is None or len(n) == 0:
            missing_normals.append(str(p.GetPath()))
    rep.add(
        "VG.027",
        not missing_normals,
        "alle Meshes haben Normalen"
        if not missing_normals
        else f"ohne Normalen: {', '.join(missing_normals[:4])}",
    )

    bad_topology = []
    for p in meshes:
        m = UsdGeom.Mesh(p)
        counts = m.GetFaceVertexCountsAttr().Get()
        idx = m.GetFaceVertexIndicesAttr().Get()
        pts = m.GetPointsAttr().Get()
        if counts is None or idx is None or pts is None:
            bad_topology.append(f"{p.GetPath()} (leer)")
            continue
        if sum(counts) != len(idx):
            bad_topology.append(f"{p.GetPath()} (counts!=indices)")
        elif max(idx) >= len(pts):
            bad_topology.append(f"{p.GetPath()} (Index out of range)")
    rep.add(
        "VG.014",
        not bad_topology,
        "Topologie konsistent"
        if not bad_topology
        else f"defekt: {', '.join(bad_topology[:3])}",
    )

    # --- Physics Bodies ----------------------------------------------------
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    rep.add("RB.001", len(rigid) > 0, f"{len(rigid)} RigidBody-Prim(s)")

    no_collider = []
    for p in rigid:
        has = any(
            c.HasAPI(UsdPhysics.CollisionAPI) for c in Usd.PrimRange(p)
        )
        if not has:
            no_collider.append(str(p.GetPath()))
    rep.add(
        "RB.COL.001",
        rigid and not no_collider,
        "jeder RigidBody hat Collider"
        if rigid and not no_collider
        else f"ohne Collider: {', '.join(no_collider) or 'kein RigidBody'}",
    )

    bad_mass = []
    for p in rigid:
        api = UsdPhysics.MassAPI(p)
        mass_attr = api.GetMassAttr()
        mass = mass_attr.Get() if mass_attr else None
        if mass is None or not math.isfinite(mass) or mass <= 0:
            bad_mass.append(f"{p.GetPath()} ({mass})")
    rep.add(
        "RB.002",
        rigid and not bad_mass,
        "Masse gesetzt"
        if rigid and not bad_mass
        else f"Masse fehlt/ungueltig: {', '.join(bad_mass) or 'kein RigidBody'}",
    )

    # --- Materialien -------------------------------------------------------
    bound = 0
    for p in meshes:
        if UsdShade.MaterialBindingAPI(p).GetDirectBindingRel().GetTargets():
            bound += 1
    colliders = [p for p in meshes if p.HasAPI(UsdPhysics.CollisionAPI)]
    rep.add(
        "MT.001",
        bound >= (len(meshes) - len(colliders)) and bound > 0,
        f"{bound}/{len(meshes)} Meshes mit Materialbindung",
    )

    # --- Semantik ----------------------------------------------------------
    root = default_prim
    cd = dict(root.GetCustomData()) if root else {}
    sem = cd.get("semantics")
    sem = dict(sem) if sem is not None else {}
    cls = sem.get("class")
    if cls is None:
        cls = cd.get("semantics:class")  # flache Schreibweise als Fallback
    rep.add("SEM.001", cls is not None, f"semantics:class = {cls}")

    return rep
