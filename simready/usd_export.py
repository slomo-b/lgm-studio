"""OpenUSD-Authoring fuer SimReady-Props.

Struktur, die hier entsteht:

    /<Name>                 Xform, RigidBodyAPI, MassAPI, Semantik
      /Looks                Scope
        /VisualMaterial     UsdPreviewSurface
        /PhysicsMaterial    UsdPhysics.MaterialAPI (Reibung)
      /Geometry
        /visual             Mesh, bindet VisualMaterial
      /Collision            Scope
        /collider_000..N    Mesh + CollisionAPI + MeshCollisionAPI(convexHull)

  Stage: metersPerUnit=1, upAxis=Z, defaultPrim=<Name>
"""

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt


def _vec3f(arr):
    return Vt.Vec3fArray([Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in arr])


def _add_mesh(stage, path, tm, purpose=None, display_color=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(_vec3f(np.asarray(tm.vertices)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces)))
    m.CreateFaceVertexIndicesAttr(
        Vt.IntArray(np.asarray(tm.faces, dtype=np.int32).reshape(-1).tolist())
    )
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

    try:
        normals = np.asarray(tm.vertex_normals, dtype=np.float64)
        m.CreateNormalsAttr(_vec3f(normals))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    except Exception as e:
        print(f"[usd] keine Normalen fuer {path}: {e}")

    lo, hi = tm.bounds
    m.CreateExtentAttr(_vec3f(np.stack([lo, hi])))

    if display_color is not None:
        # ab usd-core 24 gibt es CreatePrimvar nicht mehr direkt am Gprim
        primvar = UsdGeom.PrimvarsAPI(m).CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(_vec3f(np.tile(display_color, (len(tm.vertices), 1))))

    if purpose is not None:
        UsdGeom.Imageable(m).CreatePurposeAttr(purpose)

    return m


def _visual_material(stage, path, rgb, roughness, metallic):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path, static_friction, dynamic_friction, restitution):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(float(static_friction))
    api.CreateDynamicFrictionAttr(float(dynamic_friction))
    api.CreateRestitutionAttr(float(restitution))
    return mat


def _mass_props(tm, density):
    """Masse, Schwerpunkt, Traegheitsdiagonale -- aus Volumen und Dichte."""
    t = tm.copy()
    try:
        t.density = float(density)
        mass = float(t.mass)
        com = np.asarray(t.center_mass, dtype=np.float64)
        inertia = np.asarray(t.moment_inertia, dtype=np.float64)
        diag = np.diag(inertia).copy()
    except Exception as e:
        # nicht wasserdicht -> Volumen des Hulls als Notbehelf
        print(f"[usd] Masse aus Mesh nicht ableitbar ({e}), nutze Hull-Volumen")
        h = tm.convex_hull
        mass = float(abs(h.volume) * density)
        com = np.asarray(tm.centroid, dtype=np.float64)
        diag = np.full(3, max(mass, 1e-6) / 12.0)

    if not np.isfinite(mass) or mass <= 0:
        mass = 1.0
    if not np.all(np.isfinite(com)):
        com = np.asarray(tm.centroid, dtype=np.float64)
    diag = np.where(np.isfinite(diag) & (diag > 0), diag, max(mass, 1e-6) / 12.0)
    return mass, com, diag


def author(
    out_path,
    name,
    visual,
    colliders,
    obj_class="generic",
    material_class="generic",
    density=500.0,
    static_friction=0.8,
    dynamic_friction=0.6,
    restitution=0.1,
    roughness=0.6,
    metallic=0.0,
    approx="convexHull",
    up_axis="Z",
    annotate_semantics=True,
):
    stage = Usd.Stage.CreateNew(str(out_path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    try:
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    except Exception:
        pass
    UsdGeom.SetStageUpAxis(
        stage, UsdGeom.Tokens.z if up_axis.upper() == "Z" else UsdGeom.Tokens.y
    )

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    root_prim = root.GetPrim()
    stage.SetDefaultPrim(root_prim)

    UsdPhysics.RigidBodyAPI.Apply(root_prim)

    # --- Materialien -------------------------------------------------------
    UsdGeom.Scope.Define(stage, f"/{name}/Looks")
    mean_rgb = (
        np.asarray(visual.visual.vertex_colors[:, :3], dtype=np.float64).mean(axis=0) / 255.0
        if visual.visual.kind == "vertex"
        else np.array([0.7, 0.7, 0.7])
    )
    vmat = _visual_material(
        stage, f"/{name}/Looks/VisualMaterial", mean_rgb, roughness, metallic
    )
    pmat = _physics_material(
        stage,
        f"/{name}/Looks/PhysicsMaterial",
        static_friction,
        dynamic_friction,
        restitution,
    )

    # --- Geometrie ---------------------------------------------------------
    UsdGeom.Xform.Define(stage, f"/{name}/Geometry")
    vm = _add_mesh(stage, f"/{name}/Geometry/visual", visual, display_color=mean_rgb)
    UsdShade.MaterialBindingAPI.Apply(vm.GetPrim()).Bind(vmat)

    # --- Kollision ---------------------------------------------------------
    UsdGeom.Scope.Define(stage, f"/{name}/Collision")
    for i, part in enumerate(colliders):
        path = f"/{name}/Collision/collider_{i:03d}"
        cm = _add_mesh(stage, path, part, purpose=UsdGeom.Tokens.proxy)
        prim = cm.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_api.CreateApproximationAttr(approx)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(pmat)

    # --- Masse -------------------------------------------------------------
    mass, com, inertia = _mass_props(visual, density)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(float(mass))
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(x) for x in inertia]))

    # --- Semantik ----------------------------------------------------------
    if annotate_semantics:
        # SimReady erwartet semantische Labels. Das offizielle Schema kommt aus
        # einer Erweiterung; hier als customData in der dokumentierten Form, plus
        # assetInfo, damit die Information beim Konvertieren nicht verloren geht.
        root_prim.SetCustomDataByKey("semantics:class", obj_class)
        root_prim.SetCustomDataByKey("semantics:material", material_class)
        root_prim.SetCustomDataByKey("simready:generator", "lgm-splat2usd")
        root_prim.SetAssetInfoByKey("identifier", f"{name}_{obj_class}")

    stage.GetRootLayer().Save()
    return stage
