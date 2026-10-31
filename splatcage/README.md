# splatcage

Räumliche Schranke für Gaussian Splats. Aus Silhouetten mehrerer Ansichten wird
eine Voxelhülle gecarvt; Splats, die außerhalb liegen, fliegen raus.

Reines Rust, keine ML-Abhängigkeit, ~100 ms pro Asset.

```
cargo build --release
cargo test --release
```

## Warum das funktioniert

Ein Voxel, das in **einer** Ansicht außerhalb der Objektmaske liegt, kann nicht
zum Objekt gehören. Das ist keine Heuristik, sondern die Definition einer
Silhouette. Die Hülle ist also der Schnitt aller Sehkegel — der *Visual Hull*.

Der Preis ist bekannt und ehrlich: Konkavitäten, die keine Ansicht sieht,
füllen sich. Und: was keine Ansicht einschränkt, bleibt unbestimmt. Vier
Ansichten auf Höhe 0 bestimmen die Hochachse **nicht** — die Hülle wird dann
ein Prisma durch die ganze Box.

## Benutzen

```powershell
# Hülle aus den vier LGM-Ansichten bauen
splatcage build --lgm-dir ..\LGM\workspace_test\mv --stem catstatue_rgba `
    --res 256 --margin 1 --ply ..\LGM\workspace_test\catstatue_rgba.ply `
    --out cat.hull --obj cat_hull.obj

# Splats außerhalb entfernen
splatcage filter --hull cat.hull --ply ..\LGM\workspace_test\catstatue_rgba.ply `
    --out cat_caged.ply

# Hülle anzeigen
splatcage info --hull cat.hull
```

### Eigene Kameras (`--views`)

```json
{ "views": [
  { "mask": "v0.png", "width": 256, "height": 256,
    "fx": 280.2, "fy": 280.2, "cx": 128.0, "cy": 128.0,
    "world_to_cam": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }
] }
```

`world_to_cam` ist OpenCV-Konvention (x rechts, y unten, z nach vorn). Der Pfad
in `mask` ist relativ zur JSON-Datei.

### Masken

`--mask-mode auto` nimmt den Alphakanal, wenn er Information trägt, sonst
Helligkeit gegen weißen Hintergrund (`--threshold 240`).

## Gemessen

| Schritt | Eingabe | Zeit |
|---|---|---|
| `build` 256³, 4 Ansichten | 34k Splats | **102–116 ms** |
| `build` + `--margin 1` (Dilatation) | 256³ | +290 ms |
| `filter` | 34k Splats | **71–116 ms** |
| Hüllendatei | 256³ bit-gepackt | 2,1 MB |

Validierung: Kugel mit r = 0,35 aus 16 Ansichten ergibt 0,1781 statt 0,1796
Volumen (**99,2 %**, Diskretisierung). Zwei Unit-Tests, beide grün.

## Was der Filter bei LGM tatsächlich bewirkt

| Asset | behalten |
|---|---|
| catstatue_rgba | 96,5 % |
| bird_rgba | 95,6 % |
| maus_0451b7 | 93,6 % |
| images_a4c7d3 | 97,6 % |

Also 2–6 % entfernt. Das ist ehrlich wenig, und der Grund ist strukturell: bei
vier Ansichten auf Höhe 0 ist die Hülle praktisch die Bounding-Box des Objekts.
Der Filter räumt Streuner weg, aber er kann die Form nicht schärfen, weil die
Eingabe keine Information über oben und unten enthält.

**Der Hebel ist die Anzahl der Ansichten.** Mit 8–12 Ansichten und variierter
Elevation wird aus dem Prisma eine echte Hülle, und der Filter bekommt Zähne.
Genau das liefert der geplante Umbau des Generators (mehr Views, Elevation
gestreut) — splatcage ist darauf vorbereitet, ohne Änderung.

## Aufbau

| Datei | Inhalt |
|---|---|
| `src/hull.rs` | bit-gepackte Belegtheit, Punktabfrage, Dilatation, Box-Schnitt, Datei-I/O |
| `src/carve.rs` | Silhouetten-Carving, parallel über z-Ebenen (rayon), Early-Exit |
| `src/views.rs` | Maske, Kamera, Projektion, Orbit-Posen, JSON |
| `src/ply.rs` | generischer PLY-Leser/Schreiber, robuste Bounding-Box |

Bit-Packing statt `Vec<bool>`: 256³ sind 2,1 MB statt 16,7 MB, und die
Punktabfrage ist ein Wort-AND. Bei 34k Splats wird sie 34k Mal gemacht.
