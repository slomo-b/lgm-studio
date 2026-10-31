# Splatforge

Lokale Web-App: **ein Bild rein, ein 3D-Gaussian-Splat-Modell raus.** Läuft
komplett auf diesem Laptop — ohne NVIDIA, ohne Cloud, ohne externe Dienste.

```
Bild  ──►  LGM (ImageDream + MVDream + Gaussian-Decoder)  ──►  .ply
                                                                 │
                        ┌────────────────────────────────────────┴─────────┐
                        ▼                                                  ▼
            Browser: WebGL2-Viewer                          aceone splat-render (wgpu)
            (frei drehen, sofort)                           (Thumbnails, PNG-Export)
```

## Starten

```powershell
cd "C:\Users\Mo\Documents\splat model"
.\webui.cmd
```

Öffnet <http://127.0.0.1:7860>. Das Fenster muss offen bleiben — es ist der Server.

## Die Oberfläche

| Bereich | Funktion |
|---|---|
| **Quelle** | Bild reinziehen, klicken oder `Strg+V` einfügen. Drei Beispielbilder zum Sofort-Testen. |
| **Generieren** | Startet den Lauf. Fortschritt zeigt die vier echten Stufen (Hintergrund → Ansichten → Gaussians → Modell). |
| **Viewport** | GPU-Viewer: ziehen = drehen, scrollen = zoomen, Doppelklick = zurücksetzen. Toolbar: Auto-Rotation, Reset, Hintergrund hell/dunkel, Vollbild. |
| **Verlauf** | Alle erzeugten Modelle mit Thumbnail; klicken lädt sie in den Viewer, × löscht. |
| **Fußleiste** | Name, Splat-Anzahl, Dateigröße · PNG-Export (über aceone) · `.ply` herunterladen · Löschen. |
| **DE / EN** | Sprache umschaltbar oben rechts, inklusive aller Server-Meldungen. Wahl bleibt gespeichert. |

## Architektur

### Renderer

Es gibt **zwei** Renderer, jeder für seinen Zweck:

1. **`web/splat-viewer.js`** — eigener WebGL2-Renderer für den Browser.
   Instanzierte Billboard-Quads aus der 2D-Kovarianz (Eigenzerlegung), CPU-Sortierung
   hinten→vorne, premultiplied Alpha-Blending. Kein Radius-Clamping, keine
   Approximation durch Standbilder → beim Drehen verformen sich die Proportionen nicht.
   Läuft ohne Bibliothek, ohne CDN, ohne Worker.

2. **aceone `splat-render.exe`** (`../aceone`, Rust/wgpu) — serverseitig für
   Thumbnails im Verlauf und den PNG-Export. Wird als Prozess aufgerufen:
   `splat-render image <ply> <png> --width --height --yaw --pitch --distance --json`
   Der Adapter-String (z. B. `Intel(R) Graphics · Vulkan`) wird aus der
   JSON-Antwort gelesen und in der Kopfzeile angezeigt.

Fehlt `splat-render.exe`, läuft die App weiter: Thumbnails fallen dann auf die
Multi-View-Bilder von LGM zurück.

```powershell
# anderer Pfad zur Engine
$env:SPLAT_RENDER = "D:\pfad\zu\splat-render.exe"
```

Kamera-Umrechnung (empirisch kalibriert): `yaw = azimuth + 180`, `pitch = elevation`.
Beides in Grad.

### Server

`webui.py` — reine Standardbibliothek, kein FastAPI/Gradio. Ein HTTP-Server,
ein Inferenz-Job gleichzeitig (ein Lauf belegt die iGPU vollständig).

| Endpunkt | Zweck |
|---|---|
| `GET /api/assets` | Modelle aus `LGM/workspace_test/*.ply` + Engine-Status |
| `POST /api/generate?name=` | Bild als Roh-Body, startet Job |
| `POST /api/sample?name=` | Beispielbild starten |
| `GET /api/job` | Fortschritt (`stage`, `progress`, `msg_key`) |
| `GET /ply/<id>.ply` | Modell |
| `GET /thumb/<id>.png` | Thumbnail (aceone, gecacht) |
| `POST /api/render?id=&yaw=&pitch=&distance=` | PNG über aceone |
| `DELETE /api/assets/<id>` | Modell + Ansichten löschen |

Alle Meldungen laufen als **Keys** (`msg_key`, `error_key`) zum Client und
werden dort übersetzt — deshalb ist die Sprache vollständig umschaltbar.

## Qualität

Vier Stellschrauben wurden gemessen (gleicher Seed, gleicher Blickwinkel,
gerendert mit aceone). Alle sind über Umgebungsvariablen einstellbar.

### 1. Fast-transparente Splats verwerfen (größter sichtbarer Gewinn)

In jedem Modell sind **16–34 % der Splats fast unsichtbar** (opacity < 0,1).
Sie erzeugen den milchigen Halo ums Objekt und kosten nur Kontrast.

| Schwelle | Splats (Maus) | Wirkung |
|---|---|---|
| 0,005 (upstream) | 29 819 | Halo um das ganze Modell |
| 0,05 | 23 047 | deutlich weniger Nebel |
| **0,08 (neu)** | ~24 000 | **Halo weg, Details unverändert** |
| 0,15 | 19 778 | Kanten werden hart |
| 0,20 | 15 894 | Konturen brechen |

Gesetzt in `core/gs.py`, übersteuerbar mit `LGM_MIN_OPACITY`.

### 2. Guidance 5,0 → 6,0

Getestet an zwei Motiven (Vogel, Katzenstatue) mit festem Seed:

* **Vogel**: Sockel zeigt bei 6,0 die Holzstruktur, Flügelkanten sind getrennt
* **Katze**: Gesicht (Augen, Schnauze), Streifen und Pfote deutlich schärfer
* Bei 7,0 kommen dunkle Artefakte dazu (Schattenseite wird übersättigt)

`LGM_GUIDANCE=6.0` ist der neue Standard.

### 3. Schritte: 20 reichen

Seed-fester Vergleich 20 vs 30 Schritte: praktisch identisch, 30 ist minimal
feiner. 30 bleibt als Stufe „Fein" wählbar.

### 4. Freistellung: u2net, nicht isnet

Gegenprobe mit `isnet-general-use` (gilt als besser für allgemeine Motive):
auf dem Testbild segmentiert es **nur den Kopf** der Statue, während u2net das
ganze Objekt erfasst. u2net bleibt.

### 5. Best-of-N: mehrere Kandidaten, du wählst

Die Streuung **zwischen zwei Läufen** ist größer als der Unterschied zwischen
allen Einstellungen oben. Bei festem Seed unterscheiden sich zwei Seeds stärker
als 20 vs 30 Schritte. Also: mehrere Varianten erzeugen und die beste behalten.

In der App über **Varianten 1 / 2 / 3** (links unter „Qualität"):

* Jeder Kandidat bekommt einen anderen Seed (`1000 + i`)
* Die Kandidaten landen in `.webui/candidates/<job>/`, **nicht** im Verlauf
* Nach dem Lauf erscheint eine Leiste mit Miniaturen — anklicken, im großen
  Viewer prüfen, dann **„Diesen behalten"**
* Erst dadurch wird der Kandidat zu einem normalen Modell in `workspace_test/`
* „Verwerfen" löscht alle Kandidaten des Laufs

Zeit: 3 Varianten à 14 Schritte ≈ 105 s warm (statt ~35 s für eine).

### Was nichts gebracht hat



`bf16` statt `fp16` (2,9 s statt 2,0 s pro UNet-Forward) und höhere
Schrittzahlen (siehe oben).

## Alle Schalter

| Variable | Standard | Wirkung |
|---|---|---|
| `LGM_STEPS` | 20 | Diffusionsschritte (14 / 20 / 30) |
| `LGM_GUIDANCE` | **6.0** | Bildtreue; >7 erzeugt dunkle Artefakte |
| `LGM_MIN_OPACITY` | **0.08** | Pruning-Schwelle gegen Halo |
| `LGM_SEED` | – | fester Seed = reproduzierbar |
| `LGM_PROFILE` | – | Zeitmessung pro Phase |
| `LGM_COMPILE` | – | `unet` / `lgm` (auf dieser GPU untauglich, s. u.) |
| `SPLAT_WORKER` | 1 | `0` = Modelle nicht resident halten |
| `SPLAT_RENDER` | aceone-Pfad | andere `splat-render.exe` |



### Wo die Zeit hingeht (Profil, `LGM_PROFILE=1`)

| Phase | 30 Schritte | 20 Schritte |
|---|---|---|
| Modelle laden (LGM + ImageDream) | 11 s | 11 s |
| Hintergrund entfernen (rembg) | 0,9 s | 0,9 s |
| **Diffusion (N × UNet-Forward)** | **98,8 s** | **75,8 s** |
| Pipeline auslagern | 2,4 s | 2,4 s |
| LGM-Decoder (4 Ansichten → Gaussians) | 16–22 s | 16 s |
| `.ply` schreiben | 0,1 s | 0,1 s |
| **Gesamt (Wanduhr)** | **152 s** | **124 s** |

Ein UNet-Forward dauert **2,1 s** (Batch 10, 32×32-Latent, 942 M Parameter).
Zum Vergleich: das sind rechnerisch ~0,9 TFLOP auf einer GPU, die ~4 TFLOPS
schaffen sollte — die Zeit steckt also überwiegend in **Kernel-Launch-Overhead
und ungenutzten XMX-Einheiten**, nicht in Rechenarbeit.

Die Tabelle zeigt den alten Weg (frischer Prozess pro Bild); die Aufteilung
der Phasen gilt unverändert, nur Laden und Kompilieren fallen mit dem
Resident-Worker weg (siehe unten).

### Was gemessen wurde

| Maßnahme | Ergebnis |
|---|---|
| **Schritte 30 → 20** | **−28 s (−18 %)**, qualitativ gleichwertig (gleicher Seed, Bildvergleich) |
| Schritte → 14 | −41 s, aber sichtbar weicher (Gesicht/Details) |
| `torch.compile` (Inductor/Triton) | **untauglich**: braucht MSVC, dann >15 min Kompilierung für einen einzigen Schritt, kein Cache auf Platte |
| XPU-Graph-Capture (`torch.xpu.XPUGraph`) | **nicht unterstützt**: `UR_DEVICE_INFO_GRAPH_RECORD_AND_REPLAY_SUPPORT_EXP` fehlt im Treiber |
| bf16 statt fp16 | **langsamer** (2,9 s statt 2,0 s pro Forward) |
| Reproduzierbarkeit (`LGM_SEED=42`) | gleiche Eingabe → gleiches Ergebnis; ohne Seed schwankt die Qualität stärker als zwischen 20 und 30 Schritten |

In der App wählbar über **Qualität → Schnell (14) / Standard (20) / Fein (30)**.

### Der Resident-Worker (umgesetzt)

`webui.py` startet kein `infer.py` mehr pro Bild. Stattdessen hält der Server
eine `LgmPipeline` (in `LGM/core/pipeline.py`) im Speicher — Modelle bleiben
geladen, die GPU-Kernel bleiben kompiliert.

Gemessen, 14 Schritte, gleiche Maschine, direkt hintereinander:

| Lauf | Zeit |
|---|---|
| 1. Auftrag (kalt: Modelle laden + Kernel kompilieren) | **111 s** |
| 2. Auftrag | **52 s** |
| 3. Auftrag | **51 s** |

Mit der Standardeinstellung (20 Schritte) sind es warm **69 s** — vorher
waren es 124 s pro Bild, also **−44 %**.

Preis: der Server belegt dauerhaft ~8,5 GB RAM. Wenn die GPU anderswo
gebraucht wird:

```powershell
curl -X POST http://127.0.0.1:7860/api/unload     # Modelle freigeben
$env:SPLAT_WORKER="0"                              # gar nicht resident halten
```

Referenz: ein Modell hat 24 000–43 000 Gaussians (1,3–2,3 MB).
Thumbnail über aceone ~1 s, PNG-Export 1400×1050 ~1–2 s.

Profile selbst nachmessen:

```powershell
$env:LGM_PROFILE="1"; $env:LGM_STEPS="20"
.\run.cmd -Image LGM\data_test\catstatue_rgba.png
# -> [PROF] ... pro Phase, am Ende [PROF_JSON] mit allen Werten

.\.venv\Scripts\python.exe bench_unet.py   # nur der UNet, fp16 vs bf16, Graph-Test
```

## Installation (bereits erfolgt)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install --python .venv\Scripts\python.exe -r requirements-local.txt
.\.venv\Scripts\python.exe patch_deps.py     # transformers-Patch, idempotent
```

Pins mit Grund:

* `huggingface_hub==0.23.5` — `diffusers==0.27.2` importiert `cached_download`,
  das in neueren hub-Versionen fehlt.
* `transformers==4.40.2` — `patch_deps.py` erwartet genau diese Version.

## Warum Patches?

LGM ist CUDA-only. Angepasst (alle Stellen mit `[LOCAL PATCH]` markiert):

| Datei | Änderung |
|---|---|
| `LGM/mvdream/mv_unet.py` | `xformers` → PyTorch-SDPA, per `importlib` geladen (diffusers' Modul-Loader prüft sonst auf ein literales `import xformers`) |
| `LGM/mvdream/pipeline_mvdream.py` | `device=torch.device("cuda:0")` → vom Modul abgeleitet |
| `LGM/core/local_backend.py` | **neu** — Geräteauswahl CUDA → XPU → MPS → CPU |
| `LGM/core/xpu_patches.py` | **neu** — Causal-Mask und `nn.Embedding` auf CPU |
| `LGM/core/gs.py` | Rasterizer-Import optional, `save_ply` maskiert auf der CPU |
| `LGM/infer.py` | Backend-Wahl, UNet lokal, Pipeline-Offload in den Host-RAM, Multi-View-Dump |
| `LGM/mv_unet.py` | **neu** — Shim für Remote-Code |
| `.venv/.../transformers/models/clip/modeling_clip.py` | Pooling ohne advanced indexing (via `patch_deps.py`) |

### Der XPU-Bug

Auf dieser Plattform **hängen Index-Kernel**: `index_select`, `index_put_`,
advanced indexing — und damit auch `masked_fill_` und `nn.Embedding`.
Reproduzierbar mit torch 2.8 und 2.15; Konvolutionen, SDPA und Reduktionen
laufen normal. Symptom: ein CPU-Kern dreht durch, GPU idlet, kein Timeout.

Diagnose: `py-spy dump --pid <pid>` zeigt die Stelle sofort.

## Grenzen

* **Kein Mesh-Export.** LGM liefert Gaussians, keine Flächen. `convert.py`
  bräuchte `nvdiffrast` (CUDA) — nicht installiert.
* **Nur ein Job gleichzeitig.** Ein Lauf belegt die iGPU (~10 GB, 11,7 GB Limit).
* **PNG-Export hat schwarzen Hintergrund** — so gibt es aceone aus.
* **Modellgrenze:** LGM ist auf Objaverse trainiert (Figuren, Spielzeug, Statuen).
  Fotos von echten Tieren mit Fell oder dünnen Strukturen werden matschig.
  Ein Objekt, mittig, einfacher Hintergrund funktioniert am besten.
* Ein Lauf lastet den Rechner praktisch voll aus.

## Dateien

```
webui.py              Server (Standardbibliothek)
web/index.html        Markup
web/app.css           Gestaltung
web/app.js            Logik + Übersetzungen (DE/EN)
web/splat-viewer.js   WebGL2-Splat-Renderer
webui.cmd             Start (Execution-Policy-sicher)
run.cmd / run.ps1     Kommandozeilen-Weg ohne UI
render_preview.py     Standalone-Turntable-Renderer (PyTorch, ohne GPU)
bench_xpu.py          Backend-Benchmark (SDPA, Conv, Elementwise)
bench_unet.py         UNet-Benchmark (fp16/bf16, Graph-Capture-Test)
patch_deps.py         transformers-Patch nach venv-Neuaufbau
requirements-local.txt
SETUP.md              diese Datei
```

Ergebnisse liegen in `LGM\workspace_test\`, Caches in `.webui\` (löschbar).
