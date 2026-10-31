# LGM Studio

Natives Windows-Fenster um die bestehende Pipeline: Bild → Gaussian-Splat →
SimReady-USD. Rust für Fenster, Dateien und Prozesse; Python nur für die
Inferenz, weil Torch/XPU sich nicht nach Rust portieren lässt.

## Bauen

Voraussetzung: **MSVC-Toolchain** (Tauri unterstützt windows-gnu nicht).

```powershell
cd studio
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-msvc"
cargo build                                  # Entwicklung
npx --yes @tauri-apps/cli@2 build            # Installer (NSIS)
```

Der Installer landet in `target/release/bundle/nsis/`.

> **Leerzeichen im Pfad brechen den GNU-Build.** `windres` zerlegt
> `…\splat model\studio\target\…` am Leerzeichen. Abhilfe:
> `$env:CARGO_TARGET_DIR = "C:\lgmbuild"` vor `cargo build` setzen.

## Feinschliff

Nach der Generierung kann das Splat gegen die vier LGM-Ansichten optimiert
werden (Schalter „Feinschliff" in der Seitenleiste, ~1,5 min).

Der klassische Weg dafür ist `diff_gaussian_rasterization` — eine CUDA-Extension,
die es für die Arc-iGPU nicht gibt. `simready/diffrender.py` ersetzt sie durch
einen eigenen differenzierbaren Renderer: Beiträge nach *(Pixel, Tiefe)*
sortieren, Transmission als Segment-Summe im Lograum, Bild per `scatter_add`.
Das nutzt ausschließlich Operationen, die auf dieser XPU laufen.

Gemessen: 0,5–0,7 s pro Schritt, Katzenstatue 22,6 → 35,1 dB, Vogel 22,2 → 30,1 dB.
Das verfeinerte `.ply` bleibt LGM-kompatibel und läuft unverändert durch
`simready build`.

## Wie es zusammenhängt

```
┌─────────────── LGM Studio (Rust) ───────────────┐
│ Fenster (WebView2)   ui/index.html + app.css    │
│                             app.js              │
│ Dateidialoge          rfd                        │
│ Prozesse              std::process::Command      │
│ Download              reqwest + rustls           │
└───────────────┬─────────────────────────────────┘
                │ Kindprozess, Ausgabe zeilenweise
                ▼
        Python (.venv oder runtime/)
        LGM/infer.py  →  Gaussian-Splat .ply
        simready       →  USD + Mesh + Collider
```

Die Oberfläche zeigt **nichts Geschätztes**: jede Zeile, die Python ausgibt,
geht als Event ins Fenster. Zusätzlich schreibt jeder Lauf
`out/studio/<name>/run.log` — das Log überlebt auch einen Absturz der UI.

## Wo was liegt

| Pfad | Inhalt |
|---|---|
| `models/` | Modellgewichte, von der App selbst geladen (4,8 GB) |
| `out/studio/<name>/` | Ergebnis pro Lauf: `.ply`, `run.log`, `simready/` |
| `%LOCALAPPDATA%\LGMStudio\` | vorgesehen für die Python-Laufzeit |
| `manifest.json` | welche Modelle geladen werden (beim Bauen eingebettet) |

## Modelle

Beim ersten Start erscheint ein Einrichtungsdialog. Die App lädt 17 Dateien
(4,82 GB) von Hugging Face nach `models/`:

- `imagedream/` — UNet, Bild- und Textencoder, VAE, Configs
- `lgm/model_fp16_fixrot.safetensors` — der 3D-Generator

Eigenschaften: Fortschritt pro Datei und gesamt, Geschwindigkeit mit Restzeit,
**Wiederaufnahme** über `.part` und Range-Requests, Abbrechen, Protokoll in
`models/download.log`.

Danach läuft alles offline. Nachgewiesen mit `HF_HUB_OFFLINE=1`: Python meldet
`Loaded checkpoint from ...\models\lgm\...` und lädt keine Pipeline-Komponenten
nach.

## Python finden

Reihenfolge (erste Treffer gewinnt):

1. `LGM_STUDIO_PYTHON` (expliziter Pfad)
2. `runtime/python.exe` im Projektordner
3. `.venv\Scripts\python.exe`
4. `%LOCALAPPDATA%\LGMStudio\runtime\python.exe`

## Updates

Artefakte: `cargo tauri build` erzeugt mit `createUpdaterArtifacts: true` eine
`.sig` zum Installer. Das Manifest `latest.json` liegt **im Repo** und wird über
die Raw-URL gelesen:

```
https://raw.githubusercontent.com/slomo-b/lgm-studio/main/latest.json
```

Zwei Fallstricke, beide gemessen:

| Fallstrick | Symptom |
|---|---|
| **GitHub leitet Release-Assets um** (`releases/download/…` → anderer Host) | `error decoding response body for url (https://release-assets.githubusercontent.com/…)` — der Updater liest den Body nicht |
| **BOM im JSON** | `Set-Content -Encoding UTF8` schreibt in PowerShell 5.1 ein BOM vor die `{`; der Parser scheitert |

Deshalb: Manifest im Repo über `raw.githubusercontent.com`, und schreiben mit

```powershell
[System.IO.File]::WriteAllText($p, $json, (New-Object System.Text.UTF8Encoding($false)))
```



```
lgm-studio.exe [bild] [--auto]
```

`--auto` startet die Generierung sofort (für automatisierte Tests).
`LGM_STUDIO_AUTODOWNLOAD=1` startet den Modell-Download ohne Klick.

## Offene Punkte

| Punkt | Stand |
|---|---|
| **DPI-Skalierung** | WebView2 zeichnet mit 1,25, legt aber mit 1,0 aus → rechte Spalte abgeschnitten. Kompensation per `zoom` greift teilweise. |
| **Python-Laufzeit** | Installer lädt CPython + Wheels selbst (4,76 GB) — **scheitert aber an OOM**. Siehe unten. |
| **Updater** | Tauri-Plugin vorgesehen, Signierschlüssel und Release-URL fehlen. |

### Der OOM-Befund zur Laufzeit

Die App installiert eine portable CPython 3.11.18 (python-build-standalone) und
die Pakete per pip. Ergebnis: 4,76 GB, `torch 2.15.0+xpu`, `torch.xpu.is_available() == True`.

Beim ersten Lauf bricht die Inferenz ab:

```
RuntimeError: level_zero backend failed with error: 39
              (UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
```

Gegenprobe in derselben Minute, gleicher Befehl, gleiches Modell:

| Umgebung | Ergebnis |
|---|---|
| `runtime/` (selbst installiert) | OOM |
| `.venv/` (mit uv gebaut) | **fertig in 64,9 s** |

Die Paketstände sind identisch (`torch`, `intel-openmp`, `intel-sycl-rt`,
`onemkl-*`, `numpy`). Es liegt also an der CPython-Distribution selbst, nicht an
den Abhängigkeiten.

Ohne Wirkung versucht: `SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=0`,
`OMP_NUM_THREADS=4`, Auslagern des CLIP-Bildencoders (letzteres löst eine
XPU-Assertion aus und wurde zurückgenommen).

**Konsequenz:** Die Laufzeit nicht auf dem Zielrechner *bauen*, sondern die
funktionierende Umgebung **fertig paketieren**. Das funktioniert auch — gemessen:

| Schritt | Ergebnis |
|---|---|
| uv-CPython 3.11.16 kopieren | 72 MB |
| `.venv\Lib\site-packages` kopieren | 3 491 MB |
| **`.venv\Library` kopieren** | entscheidend! |
| `runtime\python.exe -c "import torch"` | `2.15.0+xpu`, `xpu True` |
| Generierung über `runtime\` | 145 s kalt, 72 s über die App |
| `tar -czf` | **1 773 MB** (aus 4 842 MB) |

Der Punkt, an dem es zuerst scheiterte: Die SYCL-/MKL-Bibliotheken
(`mkl_sycl_blas.6.dll`, `omptarget.sycl.wrap.dll`) liegen **nicht** in
`site-packages`, sondern in `.venv\Library\bin` — dem conda-Layout der
Intel-Pakete. Ohne diesen Ordner meldet Torch
`OSError 126: c10_xpu.dll or one of its dependencies`.

Ein selbst gebautes CPython (python-build-standalone) läuft dagegen trotz
identischer Paketstände in einen Level-Zero-OOM. Deshalb: **fertiges Archiv
ausliefern, nicht nachbauen.**


