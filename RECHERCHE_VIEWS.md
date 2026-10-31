# Mehr und bessere Views: was es fertig gibt, was wir bauen müssten

Stand 31.10.2026. Frage: LGM liefert 4 Views bei Elevation 0. Das ist die
Ursache für zwei Probleme — die Schranke wird ein Prisma (siehe
`splatcage/README.md`) und der Splat hat oben/unten keine Information.
Wie kommen wir auf 8–16 Views mit gestreuter Elevation?

---

## 1. Was das Netzwerk eigentlich braucht

Ein Multi-View-Generator ist kein 3D-Modell. Er ist ein Bildmodell, das auf
**Kameraposen als Konditionierung** trainiert wurde — bei MVDream/ImageDream
über ein MLP in die Time-Embedding, bei neueren Modellen über Ray-Embeddings
oder Kamera-Adapter. Der Unterschied zwischen „4 Views bei 0°" und „12 Views mit
Elevation" ist deshalb **keine Architektur-, sondern eine Trainingsfrage**: mit
welchen Posen wurde trainiert.

Zwei Konsequenzen:

- Man kann die Anzahl der Views nicht einfach zur Inferenzzeit hochdrehen.
  Ein auf 4 Views trainierter UNet weiß nicht, was eine fünfte ist.
- Wer Elevation will, muss mit Elevation trainieren — oder ein Modell nehmen,
  das es schon kann.

NVIDIA beschreibt das in Edify 3D ausdrücklich: *"It scales effectively, with
training on a greater number of viewpoints producing more natural and
consistent images"* — und das Modell kann zur Inferenzzeit eine beliebige Zahl
von Views sampeln, weil die Pose pro View als Embedding reingeht.

---

## 2. Was es fertig gibt

### A. View-Generatoren (das fehlende Teil)

| Modell | Was es macht | Größe / VRAM | Lizenz | Reifegrad |
|---|---|---|---|---|
| **SEVA / Stable Virtual Camera** (Stability) | 1–32 Eingabe-Views → **beliebige Zielkameras**, 14 Pfade inkl. 360°, Spirale, Lemniscate. 576p | 1,3 B | **non-commercial** | Gewichte + Code offen, HF-Space, v1.1 |
| **FrameCrafter** (CMU, ECCV 2026) | Video-DiT → permutations-invariantes NVS. **1 000 Szenen, ~1 % trainierbare Parameter** | DiT-Basis | Code offen | Code auf GitHub |
| **OrbitNVS** (Mar 2026) | NVS als Orbit-Video, Kamera-Adapter + Normalen-Zweig, +2,9 dB bei Single-View | — | Paper | wahrscheinlich ohne Gewichte |
| **MVGenMaster** (CVPR 2025) | 24 Ziel-Views aus einem Bild, 3D-Prior-gestützt | — | Paper/Code | forschungsnah |
| **CAT3D** (Google) | 8 Views intern, hochskalierbar auf 720 über mehrere Orbits | — | **keine Gewichte** | Referenz, nicht nutzbar |

### B. Konsumenten vieler Views (Rekonstruktion)

| Modell | Eingabe | Ausgabe | VRAM | Lizenz |
|---|---|---|---|---|
| **Hunyuan3D-2mv** (Tencent) | **mehrere Views** | Mesh | ~6 GB, <1 s auf A100 | MIT-artig, **aber EU/UK/Südkorea ausgeschlossen** |
| Unser TSDF (`simready/depth.py`) | beliebig viele Views + Masken | geschlossenes Mesh + Splats | CPU | eigen |
| LGM | genau 4 Views | Splats | 12 GB | MIT |

### C. Modelle, die das View-Problem gar nicht haben

| Modell | Repräsentation | VRAM | Lizenz |
|---|---|---|---|
| **TRELLIS.2-4B** (Microsoft) | Structured Latents → **Mesh + Gaussians + Radiance Field**, PBR, 512³–1536³ | 24 GB (1024er ~10 GB) | **MIT** |
| **TripoSplat** (VAST-AI) | direkt Single-Image → **Gaussian Splat**, feed-forward | klein | **MIT** |
| TRELLIS v1 | Mesh + **3D Gaussians**, Multi-Image experimentell | ≥16 GB | MIT |

Das ist die Antwort auf „warum können Mesh-Modelle auch Splats": sie sagen
nicht Splats vorher, sondern ein strukturiertes latentes Volumen, aus dem
mehrere Decoder lesen — ein Mesh-Decoder (FlexiCubes) und ein Gaussian-Decoder.
TRELLIS.2 macht genau das und ist MIT-lizenziert.

---

## 3. Drei Wege, mit Aufwand und Risiko

### Weg 1 — SEVA einsetzen, nichts trainieren
```
Bild → SEVA (8–16 Orbit-Views, Elevation gestreut) → unser TSDF → Splats
                                                   → splatcage
```
- **Aufwand:** 1 Tag Integration. Kein Training.
- **Risiko:** Lizenz **non-commercial** → für ein Produkt unbrauchbar, für
  Forschung und für den Eigengebrauch okay.
- **Technik:** braucht `flash-attn`, unter Windows also WSL. 24 GB reichen für
  sequentielles Sampling, nicht für große Batches (zu verifizieren).
- **Gewinn:** löst beide Probleme auf einmal — echte Hülle, supervidierte
  Oberseite.

### Weg 2 — eigenen Orbit-Generator per LoRA
- **Vorbild:** FrameCrafter — 1 000 Szenen, ~1 % trainierbare Parameter
  (LoRA auf Patch-Embedding und DiT), Rest eingefroren.
- **Daten haben wir schon:** Objaverse-80k-Liste (LGM) plus unseren
  Splat-/Mesh-Renderer für Orbits. Kein neues Datensatzproblem.
- **Aufwand:** 2–4 Wochen, 200–600 GPU-h ≈ 40–120 $ auf einer 3090.
- **Risiko:** mehr als „LoRA draufklatschen" — Kamera-Konditionierung muss
  sauber implementiert sein, sonst lernt das Modell nichts.
- **Gewinn:** lizenzfrei, Elevation frei wählbar, und der einzige Weg zu einem
  echten eigenen Modell.

### Weg 3 — View-Generierung ganz überspringen
TRELLIS.2 oder TripoSplat nehmen. Die machen intern, was wir mühsam
zusammenbauen: latentes Volumen → Mesh **und** Gaussians.
- **Aufwand:** 1–2 Tage Integration, 24 GB VRAM.
- **Risiko:** keins lizenzseitig (MIT). Aber: es ist nicht mehr unser Modell,
  und die SimReady-Schicht müssen wir trotzdem selbst bauen.
- **Gewinn:** sofort Spitzenqualität statt Bastelei.

---

## 4. Die Lizenz-Falle

Zwei Punkte, die die Auswahl faktisch entscheiden:

1. **Hunyuan3D** (2.0/2.1/2mv/HY-World) schließt die **EU, UK und Südkorea**
   per Lizenz aus. Wenn du in Deutschland sitzt, ist das ein Blocker, kein
   Detail.
2. **SEVA** ist **non-commercial**. Für Prototyp und Forschung gut, für ein
   Produkt nicht.

Sauber bleiben **TRELLIS.2** und **TripoSplat** (beide MIT), plus alles, was wir
selbst trainieren.

---

## 5. Empfehlung

**Kurzfristig Weg 1, strategisch Weg 2, und Weg 3 als Messlatte.**

Konkret:

1. **SEVA hinstellen und messen** (1 Tag, kostenlos, non-commercial okay für
   diesen Schritt). Acht Views bei −20°/0°/+20° → durch unseren TSDF → durch
   `splatcage`. Das liefert zum ersten Mal eine echte Hülle und zeigt, wie viel
   die View-Zahl wirklich bringt.
2. **TRELLIS.2 als Referenz danebenstellen** (1 Tag). Nicht integrieren,
   sondern *messen*: wie gut ist ein MIT-Modell auf denselben Bildern? Wenn es
   unseren Aufbau schlägt, ist die Frage beantwortet, ob wir überhaupt um LGM
   herum bauen sollten.
3. **Erst danach entscheiden**, ob Weg 2 (LoRA, 2–4 Wochen) sich lohnt. Wenn
   TRELLIS.2 schon besser ist als alles, was wir mit LGM + 8 Views erreichen,
   wäre der LoRA-Aufwand verlorene Liebe.

Die Reihenfolge ist wichtig: **erst messen, was es gibt, dann bauen, was fehlt.**
Wir haben in diesem Projekt schon zweimal eine Woche in eine Richtung investiert,
deren Grundannahme sich danach als falsch erwies (Poisson, LGM-Geometrie).

---

## 6. Was von unserer Arbeit bleibt

Unabhängig vom Weg:

| Baustein | bleibt gültig, weil |
|---|---|
| `splatcage` (Rust) | braucht nur Masken + Posen, egal woher die Views kommen |
| `simready/` (USD, Physik, Lint) | Modell-agnostisch, nimmt jedes Mesh |
| TSDF in `simready/depth.py` | nimmt beliebig viele Views, kein Training |
| UI aus LGM Studio | Pipeline-Schicht, unabhängig vom Generator |

Das ist der Grund, warum die Arbeit an der Schranke und der SimReady-Schicht
nicht verloren ist, auch wenn der Generator ausgetauscht wird.

---

## 7. Nachtrag: der billigste Weg lag schon im Code

`mvdream/mv_unet.py:37` — `get_camera(num_frames, elevation=0, …)`. Die Elevation
ist **kein** freier Parameter der Pipeline, sie geht als Kamera-Embedding ins
Modell. Und die Pipeline reicht sie durch:

```python
mv_image = pipe('', image, guidance_scale=5.0,
                num_inference_steps=30, elevation=0)   # <- war immer 0
```

`infer.py` hat die Elevation schlicht nie benutzt. Gemessen
(`tools/elev_probe.py`, 20 Schritte, gleicher Seed):

| Durchlauf | Zeit | Ergebnis |
|---|---|---|
| elev −25° | 58,0 s | Blick von unten |
| elev 0° | 55,7 s | Augenhöhe |
| elev +25° | 63,0 s | Blick von oben, Rücken und Standfläche sichtbar |

Mittlere Pixelabweichung −25°↔0°: 0,060 · −25°↔+25°: 0,089. Die Ansichten
ändern sich strukturell, nicht nur im Rauschen. **Drei Durchläufe = 12 Views mit
vertikaler Abdeckung, 176 s auf der iGPU, ohne Training, MIT-Lizenz.**

Damit ist die Begründung für SEVA weggefallen: wir brauchen keinen 1,3-B-Prior,
wenn der vorhandene 415-M-Generator die Höhe schon kann. Der Preis ist Zeit
(3× Diffusion) und die Inkonsistenz zwischen den Durchläufen — jedes Bündel
sieht das Objekt leicht anders.

### Der Test dazu, und er ist negativ

12 Views aus drei Durchläufen (`tools/gen_views_elev.py`, 172 s), durch
`splatcage`:

| Ansichten | belegt | größte Komponente |
|---|---|---|
| nur elev 0° (4 Views) | 4,71 % | 95,6 % |
| nur elev −25° (4 Views) | 2,98 % | 98,5 % |
| nur elev +25° (4 Views) | 2,55 % | 96,2 % |
| **alle 12 zusammen** | **0,29 %** | 71,6 % |

Jedes Bündel allein ist kohärent. Die Vereinigung ist **zehnmal kleiner als
jedes einzelne Bündel** — die drei Durchläufe schneiden sich gegenseitig weg.
Ein Vorzeichenfehler ist es nicht: Umdrehen der Elevation macht es schlechter
(0,24 % statt 0,43 %).

**Fazit: die drei „Kater" sind drei verschiedene Kater.** Das Modell erzeugt
jeden Durchlauf unabhängig; die Ansichten sind untereinander nicht konsistent
genug, um sich zu ergänzen. Der billige Weg über mehr Elevation funktioniert
also nicht.

Nebenbefund, der viel Zeit gekostet hat: bei Kameraabstand 1,5 und 49° Blickfeld
ist am Objekt nur ±0,68 sichtbar. Eine größere Carving-Box wird in den
Randbereichen **nie** durchstochen, und die Hülle wandert in eine Ecke. Der
Fehler ist jetzt als Warnung in `splatcage` eingebaut.

## 8. Kleine offene Modelle, die es schon gibt

| Modell | Größe | Ausgabe | Lizenz | Lokal? |
|---|---|---|---|---|
| **TripoSplat** | ~3,1 GB fp16 (Encoder 1 749 + Denoiser 745 + Decoder 476 + VAE 69 + Octree 102 MB); int4-Varianten | **direkt Gaussians**, 32k–262k, 20 Schritte | **MIT** | **ja** — es läuft bereits im Browser über ONNX/int4, also erst recht auf einer iGPU |
| **Pixal3D** | ~4B-Klasse, `--low_vram` auf 1024 | Mesh (GLB) | MIT, **aber `extra_gated_eu_disallowed`** | nein, EU gesperrt |
| **InstantMesh** | Zero123++ (SD-basiert) + LRM | 6 Views → Mesh, ~10 s | Apache/MIT | grenzwertig, ~1 B |
| **Unique3D** | SD-basiert + Normalen-Diffusion | 4 orthogonale Views @256 → Mesh | offen | grenzwertig |
| **MVDream** | SD 2.1 | 4 Views @256, **Elevation konditioniert** | MIT | ja (dieselbe Klasse wie ImageDream) |
| TRELLIS.2 | 4 B | Mesh + Gaussians + PBR | MIT | nein (24 GB) |
| SEVA | 1,3 B, 576p | bis 32 Views | non-commercial | nein |

**TripoSplat ist der ernsthafte Kandidat für „top und lokal":** MIT, ein Bild rein,
Gaussians raus, und der Browser-Port beweist, dass die Rechnung auf schwacher
Hardware aufgeht. Es löst das View-Problem allerdings durch Ignorieren — es sagt
Splats direkt vorher, ohne Zwischenansichten.

## Quellen

- SEVA / Stable Virtual Camera, ICCV 2025:
  <https://stable-virtual-camera.github.io/> · <https://huggingface.co/stabilityai/stable-virtual-camera>
- FrameCrafter, ECCV 2026 (1K Szenen, ~1 % Parameter):
  <https://frame-crafter.github.io/>
- OrbitNVS, arXiv 2603.19613: <https://arxiv.org/abs/2603.19613>
- MVGenMaster, CVPR 2025 (24 Views)
- CAT3D, NeurIPS 2024: <https://cat3d.github.io/>
- TRELLIS.2-4B (MIT, Mesh + Gaussians): <https://huggingface.co/microsoft/TRELLIS.2-4B>
- TripoSplat (MIT, direkt Splats): <https://huggingface.co/VAST-AI/TripoSplat>
- Hunyuan3D-2mv (Multi-View-Eingabe, Lizenzechranken):
  <https://github.com/Tencent-Hunyuan/Hunyuan3D-2>
