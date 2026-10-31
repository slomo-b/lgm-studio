# LGM: alle Hebel für bessere Assets

Stand 31.10.2026. Getrennt nach **ohne Training**, **kleines Training**,
**Architektur** und **Backbone**. Wo möglich mit gemessenen Zahlen aus diesem
Projekt oder aus Papers mit Vergleich gegen LGM.

---

## 0. Was LGM heute limitiert (gemessen an unserem Code)

| Eigenschaft | Wert | Folge |
|---|---|---|
| Parameter | 415 M | klein; die Qualität kommt aus dem Prior |
| Gaussians | 65 536 (4 × 128²) | Detaildecke: dünne Strukturen fehlen |
| Farbmodell | **SH-Grad 0** (`gs.py:81`) | keine view-dependent Farbe, kein Glanz |
| Positions-Aktivierung | `clamp(-1,1)` (`models.py:40`) | Gradient außerhalb null, Gaussians kleben an den Würfelflächen |
| Fusion der Views | **Konkatenation** (`models.py:107`) | keine 3D-Fusion, Nähte, Redundanz |
| Verlust | MSE + LPIPS auf Pixeln | keine Geometrie-Supervision |
| Eingabe | 4 Views @ Elevation 0 | Oben/Unten unsupervidiert |

Diese sechs Punkte sind die Ursachen. Alles Folgende greift einen davon an.

---

## 1. Ohne Training — sofort umsetzbar

### 1.1 Predict-then-Refine ← **der größte Hebel**

Der Konsens der Literatur 2025/26 ist eindeutig: Feed-Forward-Ausgaben sind
eine *Initialisierung*, keine Endqualität. Alle neueren Arbeiten bauen darauf:

| Arbeit | Aussage |
|---|---|
| **ForeSplat** (2026) | „predict-then-refine": Feed-Forward als Startpunkt für ein bescheidenes Optimierungsbudget |
| **ReSplat** (ICLR 2026) | gelernte iterative Verfeinerung, Delta-Gaussians |
| **GIFSplat** (CVPR 2026) | iterative Verfeinerung ohne Test-Time-Gradienten, **+2,1 dB PSNR** |
| **Diff3R** (2026) | differenzierbare 3DGS-Optimierungsschicht, modellagnostisch |

Konkret für uns: die vier generierten Ansichten als Supervision nehmen und
**200–500 Schritte klassisches 3DGS** auf dem LGM-Splat rechnen. Das korrigiert
Geometrie und Farbe gegen echte Beobachtungen statt gegen den Prior.

**Einschränkung, ehrlich:** das braucht `diff_gaussian_rasterization`, also
CUDA. Auf der iGPU geht es nicht — auf einer gemieteten Karte kostet es
Sekunden bis wenige Minuten pro Asset (Cent-Betrag).

#### Umsetzung ohne CUDA — gemessen 31.10.2026

Der Rasterizer ist ersetzbar. `simready/diffrender.py` komponiert stattdessen
vektorisiert: nach *(Pixel, Tiefe)* sortieren, Transmission als Segment-Summe im
Lograum, Beiträge per `scatter_add` aufs Bild. Das nutzt nur Operationen, die auf
dieser XPU laufen (`index_add_` hängt, `sort`/`cumsum`/`gather`/`scatter_add`
nicht — geprüft in `tools/xpu_opcheck.py`).

| | Wert |
|---|---|
| Schrittkosten | 0,5–0,7 s (vorwärts + rückwärts, 33 k Gaussians, 256²) |
| 150 Schritte | ~1,5 min pro Asset |
| Katzenstatue | 22,6 → 35,1 dB gegen die vier Ansichten (**+12,5 dB**) |
| Vogel | 22,2 → 30,1 dB in 40 Schritten (**+8,0 dB**) |
| SimReady danach | 12/12 Regeln, 26 Meshes, 41,7 s |

Was der Feinschliff *nicht* löst: in Zwischenansichten entstehen einzelne
Sprenkel, weil vier Ansichten keine 360-Grad-Abdeckung sind. Bewegung auf den
Ausgangspunkt (`--reg-shift`) und Farbanker (`--reg-color`) dämpfen das, beseitigen
es aber nicht. Wer das sauber will, braucht mehr Ansichten, nicht mehr Schritte.

### 1.2 SH-Grad erhöhen

`gs.py` rendert mit `sh_degree=0`. Grad 1–2 kostet 3 bzw. 8 Koeffizienten pro
Kanal und bringt Glanz und Blickwinkelabhängigkeit. Ohne Neusatz der Gewichte
nicht möglich (das Modell sagt nur DC vorher) — **aber** es ist ein billiger
Zusatz im Refine-Schritt: dort werden SH-Koeffizienten ohnehin optimiert.

### 1.3 Aktivierung reparieren

`pos_act = clamp(-1,1)` → `tanh`. Kein Training nötig, um den *Fehler* zu
benennen, aber ein Finetune, um ihn zu nutzen. Bei jedem künftigen Training
mitnehmen.

### 1.4 Anti-Aliasing und Aufräumen

- **Mip-Splatting**-Filter (3D-Glättung nach Bildschirmgröße) gegen Flimmern
  und Splitter beim Herauszoomen — als Nachbearbeitung anwendbar.
- **Opacity-/Scale-Regularisierung + Pruning**: Gaussians unter einer
  Transparenzschwelle entfernen. Unsere Schranke (`splatcage`) macht das
  räumlich, Pruning macht es statistisch.

### 1.5 Die billigen Regler, die wir schon gemessen haben

| Regler | Messung |
|---|---|
| 16 statt 20 Schritte | gleiche Qualität, −20 % Zeit |
| Elevation variieren | **geht nicht** — Durchläufe sind inkonsistent (gemessen) |
| CFG 5 → 6 | im Repo bereits als Default gesetzt, ungeprüft |

---

## 2. Kleines Training (Tage, eine 3090)

### 2.1 Geometrie-Supervision
Depth- und Normalen-Kopf als Auxiliary Task. Die Ground Truth fällt beim
Bauen des Datensatzes praktisch gratis an (aus 3DGS-Renders). Das ist der
einzige Weg, das Modell *Oberflächen* statt *Pixel* lernen zu lassen.

### 2.2 Auflösung
`splat_size` 128 → 192/256: 65 k → 147 k/262 k Gaussians. Der Decoder wächst
quadratisch, 24 GB reichen mit Batch 2–4 und Gradient-Accumulation.

### 2.3 Fusion statt Konkatenation
Die vier Ebenen in einen gemeinsamen 3D-Raum heben (sparse 3D-Conv, Voxel-
Pooling, Cross-View-Attention). Der Bauplan liegt im Repo: `mvdream/mv_unet.py`
hat mit `BasicTransformerBlock3D` bereits View-Attention.

---

## 3. Architektur — was die Nachfolger messbar besser machen

### 3.1 Per-Pixel-Gaussians statt 4 Ebenen (GS-LRM)

**GS-LRM: +8 dB PSNR gegen LGM** auf GSO und ABO, bei praktisch gleichem
Trainingsaufwand (LGM: 32×A100, 4 Tage; GS-LRM low-res: 64×A100, 2 Tage).

Der Unterschied ist strukturell: GS-LRM sagt **pro Pixel** eine Gaussian vorher
aus einem Multi-View-Transformer. Damit skaliert die Gaussian-Zahl mit der
Eingabeauflösung — bei LGM ist sie auf 4 × 128² **festgenagelt**. Genau das
erklärt die Detaildecke.

### 3.2 Geometrie und Erscheinung trennen (2Xplat)

**2Xplat: 23,76 gegen 20,38 PSNR** (+3,4 dB) gegen ein monolithisches Modell,
gleiche Geometrie-Backbone, 12 Eingabe-Views auf DL3DV.

Die Einsicht: wer Geometrie und Aussehen in einem Netz vermischt, bekommt
beides mittelmäßig. Zwei Experten — einer für Form, einer für Farbe — schlagen
das deutlich. Für uns relevant, weil **unser** Fehlerbild genau das ist:
brauchbare Silhouette, schlechte Oberfläche.

### 3.3 Auflösung nachträglich (SR3R, LGTM, SubSplat)

- **SR3R** (CVPR 2026): plug-and-play 3D-Super-Resolution, nimmt ein
  Feed-Forward-3DGS als Gerüst und skaliert es hoch
- **LGTM** (2026): 4K, „less Gaussians, texture more" — 1,8× Speicher für 64×
  Pixel, weil Aussehen nicht in Gaussians, sondern in Textur geht
- **SubSplat**: Sub-Pixel-Reparametrisierung für hochauflösende pixel-aligned
  Modelle

Das ist der Pfad, wenn die Gaussian-Zahl der Deckel ist — ohne das ganze Modell
neu zu trainieren.

### 3.4 Eingabe-View-Auswahl (Flex3D)
Flex3D kuratiert die Eingabeansichten statt sie fest zu verdrahten. Für uns
heißt das: nicht blind Elevation 0 nehmen, sondern die Ansichten wählen, die
das Objekt am besten abdecken.

---

## 4. Backbone tauschen

TRELLIS.2 (MIT, Mesh + Gaussians, 4 B) und TripoSplat (MIT, ~3 GB, direkt
Gaussians) sind heute stärker als LGM. Sie lösen das Problem anders: nicht
mehr Ansichten, sondern ein latentes Volumen mit mehreren Decodern.

Ehrlich dazu: das ist kein „LGM verbessern", sondern ein Wechsel.

---

## 5. Rangfolge nach Hebel pro Aufwand

| # | Maßnahme | Erwartung | Aufwand | Braucht |
|---|---|---|---|---|
| 1 | **Predict-then-Refine** (200–500 3DGS-Schritte) | größter Einzelgewinn, Literatur +2 dB | 1 Tag | ~~CUDA~~ **erledigt, ohne CUDA** |
| 2 | Pruning + Anti-Aliasing + Schranke | weniger Splitter, saubere Kanten | Stunden | lokal |
| 3 | 16 Schritte, CFG prüfen | −20 % Zeit | Minuten | lokal |
| 4 | Depth/Normal-Supervision + tanh | bessere Oberflächen | Tage | 3090 |
| 5 | Fusion der Views | Nähte weg | Wochen | 3090 |
| 6 | Per-Pixel-Gaussians (GS-LRM-Ansatz) | +8 dB in der Literatur | Wochen–Monate | Cluster |
| 7 | Backbone-Wechsel | Sprung | Tage | 24 GB+ |

**Punkt 1 ist der einzige, der allein durch Rechnen besser wird, ohne ein
einziges Gewicht anzufassen.** Und er passt zu unserem Aufbau: die vier
generierten Ansichten sind bereits da, die Schranke liefert eine saubere
Initialisierung, und Pruning entfernt vorher die Streuner.

---

## Quellen

- LGM, ECCV 2024 Oral: <https://arxiv.org/abs/2402.05054>
- GS-LRM (+8 dB): <https://sai-bi.github.io/project/gs-lrm>
- 2Xplat (Geometrie/Aussehen getrennt): <https://arxiv.org/html/2603.21064v3>
- ForeSplat (predict-then-refine): <https://arxiv.org/html/2605.22020v2>
- GIFSplat (iterativ, +2,1 dB): <https://cvpr.thecvf.com/virtual/2026/poster/36510>
- ReSplat (lernen zu optimieren): <https://haofeixu.github.io/slides/20260309_resplat.pdf>
- Diff3R (differenzierbare Optimierungsschicht): <https://www.youtube.com/watch?v=IxzNSAdUY70>
- LGTM (4K, Textur statt Gaussians): <https://arxiv.org/html/2603.25745v1>
- SR3R (plug-and-play 3D-SR): <https://cvpr.thecvf.com/virtual/2026/poster/39398>
- Übersicht: <https://github.com/ziplab/Awesome-Feed-Forward-3D>
