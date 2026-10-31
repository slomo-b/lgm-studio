# LGM seziert — und was für SimReady-Assets fehlt

Ziel: aus einem Bild ein **SimReady**-Asset machen (OpenUSD, Isaac Sim / PhysX,
kollidierbar, metrisch korrekt). Dieses Dokument trennt, was LGM selbst
limitiert, von dem, was gar nicht im Modell steckt.

Alle Zahlen sind an diesem Repo gemessen (`_paramcount.py`, Config `big`).

---

## 1. Ist-Zustand: was LGM wirklich ist

| Größe | Wert |
|---|---|
| Parameter gesamt | **415,0 M** (0,83 GB fp16) |
| davon UNet-Encoder / Mid / Decoder | 114,0 M / 42,0 M / 259,1 M |
| Gaussians pro Asset | **65 536** (4 Views × 128²) |
| Eingabe | 4 RGB-Views à 256² + Plücker-Ray-Embedding (9 Kanäle) |
| Ausgabe | 4 Ebenen à 14×128×128 → Positions-/Opacity-/Scale-/Rot-/RGB-Kanäle |
| Supervision | MSE auf RGB + Alpha, plus LPIPS |

**Die entscheidende Einordnung: der Generator ist klein.** 415 M Parameter sind
weniger als ein Achtel dessen, was man bei „Diffusionsmodell für 3D" erwartet.
Die Bildqualität kommt fast vollständig aus dem **eingefrorenen** Multiview-Prior
(ImageDream, 256², 30 Schritte). Für alles, was LGM an Bildschärfe fehlt, ist
zuerst dieser Prior verantwortlich — nicht der 3D-Teil.

### Ablauf bei der Inferenz

```
Bild → rembg (Alpha) → recenter(border 0.2)
     → ImageDream, 4 Views @ elevation 0, CFG 5.0, 30 Steps, 256²
     → Plücker-Rays
     → LGM-UNet (2D!, 9→14 Kanäle, 256→128)  ⇒ 65 536 Gaussians
     → PLY (pos ∈ [-1,1]³)
```

---

## 2. Sechs strukturelle Befunde

### B1 — Es gibt keine 3D-Fusion. Null.

`core/models.py:93` faltet die vier Ansichten in die **Batch-Dimension**
(`images.view(B*V, C, H, W)`), `:98` formt zurück, `:107` flacht zu einer
einzigen Gaussian-Menge ab.

Damit sagt jede Ansicht **unabhängig** ihre eigenen 16 384 Gaussians vorher, und
die vier Mengen werden schlicht **konkateniert**. Es gibt keine 3D-Faltung,
kein Voxel-Pooling, keine Cross-View-Attention. Das einzige Multi-View-Signal
sind die Plücker-Rays pro Ansicht.

Folgen: doppelte/überlappende Gaussians an den View-Grenzen, sichtbare Nähte,
keine globale Konsistenz, „aufgeblasene" Rückseiten. **Das ist der größte
einzelne Architekturfehler** — und gleichzeitig der am besten reparierbare.

### B2 — Ansichtsunabhängige Farbe (SH-Grad 0)

`core/gs.py:81` setzt `sh_degree=0`, `:94` speist RGB direkt als
`colors_precomp` ein. Es gibt keine view-dependent Farbe, also kein
Glanzverhalten. Für Physik-Assets zweitrangig, für Optik/Detektion relevant.

### B3 — Hartes Clamping der Positionen

`core/models.py:40`: `pos_act = x.clamp(-1, 1)`. Außerhalb von [-1,1]³ ist der
Gradient exakt null, innerhalb ungebremst. Gaussians „kleben" an den
Würfelflächen; das erzeugt die typischen flachen Sockel und Decken. `tanh`
oder ein weicher Clamp wäre korrekt.

### B4 — Fester Prior bei elevation 0

`infer.py:110`: `elevation=0`, vier Ansichten bei 0°/90°/180°/270°. Oben und
unten sind damit praktisch **unsupervidiert**. Für Robotik, die von schräg oben
greift, ist das genau die falsche Voreinstellung.

### B5 — Das Optimierungsziel kennt keine Geometrie

`core/models.py:150`: `MSE(render, GT) + MSE(alpha, mask)`, `:162` plus LPIPS.
Alles bildraumbasiert. Es gibt **keinen** Tiefen-, Normalen-, SDF- oder
Chamfer-Term. Das Modell wird für *richtig aussehende Pixel* trainiert, nicht für
*korrekte Oberflächen*. Ein Asset, das kollidiert statt gerendert zu werden,
braucht das Gegenteil.

### B6 — Auflösungsdeckel

65 536 Gaussians, also 4 × 128 × 128. Dünne Strukturen — Kabel, Henkel,
Scharniere, Gitter — sind strukturell nicht darstellbar. Ein Regal wird zur
Platte, ein Stuhl verliert die Streben.

---

## 3. Der eigentliche Gap: SimReady ≠ Splat

SimReady Foundation prüft maschinell gegen Regeln mit IDs. Was LGM liefert,
passt in keine davon:

| SimReady verlangt | Regel | LGM liefert |
|---|---|---|
| `metersPerUnit = 1.0`, `upAxis = Z` | UN.006 u. a. | PLY in [-1,1]³, keine Einheit, keine Achse |
| Mesh mit Normalen, Winding, sauberer Topologie | VG.027 | Gaussians, **kein Mesh** |
| `CollisionAPI` auf jedem Rigid Shape | RB.COL.001 | nichts |
| `UsdPhysics` RigidBody, Masse, Inertia | RB.001 | nichts |
| `UsdShade` Material + Bindings, Physikmaterial (Reibung) | — | nur RGB, kein PBR |
| Semantische Labels (Klasse, Materialtyp) | — | nichts |
| `defaultPrim`, stabile Prim-Pfade | — | nichts |

**Konsequenz:** „LGM für SimReady verbessern" ist zu grob **70 % Pipeline-Bau
und 30 % Modellarbeit**. Wer am Modell feilt, bevor die Pipeline steht, optimiert
ein Asset, das niemand laden kann.

---

## 4. Plan, nach Hebel sortiert

### Stufe A — Pipeline (kein Training, größter Hebel)

| Schritt | Werkzeug | Ergebnis |
|---|---|---|
| A1 Splat → Mesh | 2DGS / SuGaR / GOF oder Depth-basiert | Dreiecke statt Punkte |
| A2 Reparatur | Open3D, pymeshlab | wasserdicht, Normalen, UV, LOD |
| A3 Maß & Pose | Klassen-Prior per VLM/CLIP + Silhouette | Meter, Pivot am Fußpunkt, upAxis Z |
| A4 Collider | CoACD / Convex Decomposition | Kollisionskörper |
| A5 Physik + Labels | `usd-core` (pxr) | Masse, Reibung, Semantik |
| A6 Validierung | `simready-foundation` Rules | Report mit Regel-IDs |

Aufwand: 1–2 Tage, überwiegend CPU, lokal testbar. Danach lädt Isaac Sim das
Asset und es kollidiert.

### Stufe B — Modell, billig (Training auf gemieteter 3090)

- **B1** `splat_size` 128 → 192/256 (Gaussians 65 k → 147 k/262 k). Decoder wächst
  quadratisch; 24 GB reichen mit Batch 2–4 + Gradient-Accumulation.
- **B2** `pos_act` → `tanh`, `conv` entfernen (steht so im Code), `scale_act`
  kalibrieren.
- **B3** **Geometrie-Supervision**: Depth- und Normalen-Kopf, GT direkt aus den
  gebackenen Splats. Das ist der Punkt, an dem der Splat-Datensatz echten Wert
  hat — er liefert Tiefe und Normalen **gratis**, Objaverse-Renders nicht.

### Stufe C — Modell, architektonisch (der eigentliche LGM-Fix)

- **C1** Cross-View-Fusion statt Konkatenation (B1 oben): die vier
  Gaussian-Mengen in einem gemeinsamen 3D-Raum zusammenführen — sparse 3D-Conv,
  Voxel-Pooling oder Attention über die Views. Behebt Nähte und Redundanz.
- **C2** 4 → 6/8 Input-Views mit variierter Elevation, damit die Oberseite
  supervidiert wird.
- **C3** Aux-Head für kanonische Pose (Up-Vektor), damit A3 nicht raten muss.

### Stufe D — Backbone (dann ist es nicht mehr LGM)

Neuerer Multiview-Prior oder ein natives 3D-Generativmodell (TRELLIS,
Hunyuan3D-2). Höhere Decke, aber Verlust der bestehenden Codebasis — und
mindestens 24–48 GB VRAM für Training.

---

## 5. Empfehlung

**A zuerst, dann B3 + C1 gemeinsam.**

Begründung: Stufe A macht aus LGM überhaupt erst einen Asset-Generator und ist
lokal und kostenlos prüfbar. B3 und C1 sind danach die zwei Änderungen, die das
Modell messbar verbessern statt nur anders aussehen zu lassen — und sie passen
zum Datensatz-Plan: gebackene Splats liefern die Tiefe, die LGM heute fehlt.

Reihenfolge, wenn es ein Produkt werden soll:
`A1 → A3 → A4 → A6` (lauffähig), dann `B3 → C1` (besser), dann `A2/A5` (schön).

Kosten Modellstufen: eine 3090 liegt bei 0,17–0,24 $/h; ein Trainingslauf über
Nacht kostet 4–8 $.

---

## 6. Stufe A ist gebaut — Stand und Messwerte

Eigenes Paket `simready/`, eigene Umgebung `.venv-simready` (die LGM-Venv bleibt
unangetastet).

```
python -m simready build <splat.ply> -o out/<name> \
       --obj-class mug --target-size 0.09 --preview
```

Schritte: Splat → Punktwolke → TSDF-Netz → Metrik/Pivot → Collider → USD → Lint.
Alles CPU, keine GPU.

### Gemessen über 13 Assets aus `LGM/workspace_test`

| Kennzahl | Wert |
|---|---|
| Zeit pro Asset | **15,5 s** (9,0–24,4 s) |
| wasserdicht | **13/13** |
| SimReady-Lint | **13/13 mit 12/12 Regeln** |
| Ausgabe | `.usda` (~5 MB), `.obj`, `build_report.json` |

### Warum TSDF und nicht Poisson

Poisson braucht verlässliche Normalen. Ein 3DGS liefert die nicht: LGM
konkateniert vier Ansichts-Schalen, deren Oberflächen sich um Millimeter
widersprechen. Messwerte: Poisson-Ergebnis nie wasserdicht, Loch im Rumpf,
ausgefranste Ränder.

TSDF braucht **keine** Normalen. Fünf Fehler mussten dafür gefunden werden:

| Fehler | Ursache |
|---|---|
| 0 Faces vom Mesher | Open3Ds `ScalableTSDFVolume`/`UniformTSDFVolume` integriert in dieser Installation nichts (mit synthetischer Kugel verifiziert) → eigener TSDF in numpy |
| Volumen 10× zu klein | Löcher im Z-Buffer stanzen Tunnel durchs Volumen |
| Volumen immer noch 6× zu klein | An Löchern sieht der Z-Buffer die **Rückseite** (1,9 statt 1,1) → Fliegende-Pixel-Filter |
| Volumen 3,7 % statt 24 % | Punktabstand 1,4 px, Einzelpixel-Z-Buffer löchrig → Punkte als Scheiben rastern |
| Objekt auf dem Kopf | Y→Z-Rotation mit falschem Vorzeichen (−90° statt +90° um X) |

### Grenzen

- **Visual Hull**: Konkavitäten laufen zu (Kugel +13 % Volumen). Für Kollision
  unproblematisch, für Optik sichtbar. Freiraum-Carving ist implementiert, aber
  durch die Rückseiten-Leckage noch nicht aktiv.
- **Skalierung ist ein Prior, keine Messung.** `--target-size` oder Klassenprior.
- **Kein PBR.** Vertexfarben plus `UsdPreviewSurface`; keine Textur, kein
  Metall/Rauheit aus dem Bild.
- Collider ist ein konvexer Hull. CoACD braucht 8+ Minuten pro Asset und lohnt
  erst bei echt konkaven Formen (`--collider coacd`).

