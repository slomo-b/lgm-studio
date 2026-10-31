# LGM Studio

Bild rein, 3D-Asset raus: aus einem Foto entsteht ein Gaussian-Splat, der auf
seine Silhouette begrenzt und als SimReady-USD exportiert wird.

Native Windows-App (Tauri 2, Rust) mit Python nur dort, wo es nicht anders
geht — der Inferenz. Alles Sichtbare, alle Dateien und alle Prozesse laufen in
Rust.

## Was funktioniert

| Baustein | Stand |
|---|---|
| Bild → Gaussian-Splat über die Oberfläche | ✅ 72 s pro Asset (16 Schritte) |
| Modelle laden sich selbst (4,8 GB, Hugging Face) | ✅ offline danach, `models/download.log` |
| Räumliche Schranke (`splatcage`) | ✅ 256³ in ~100 ms, Silhouetten-Carving |
| Splat → Mesh → SimReady-USD | ✅ 13/13 Assets valide, 12/12 Regeln |
| Installer | ✅ 1,71 MB NSIS, Startmenü, Deinstaller |
| Auto-Updater | ⚠️ Artefakte und Endpunkt stehen, Prüfung im Client fehlt |

## Was nicht funktioniert

**Die Python-Laufzeit** lässt sich nicht auf dem Zielrechner bauen: ein selbst
kompiliertes CPython (python-build-standalone) läuft trotz identischer
Paketstände in einen Level-Zero-OOM. Die funktionierende Umgebung muss
**gepackt ausgeliefert** werden (4,84 GB → 1,77 GB Archiv). Das Rezept steht in
[`studio/README.md`](studio/README.md).

**Die DPI-Skalierung**: WebView2 legt den Viewport ~1,27× breiter an als das
Fenster. Das Layout hält den rechten Rand deshalb frei — funktional, aber nicht
schön.

## Aufbau

```
studio/      Tauri-App: Fenster, Dialoge, Downloader, Prozesssteuerung
splatcage/   Rust-Crate: Voxelhülle aus Silhouetten, Splats darauf begrenzen
simready/    Splat -> TSDF-Mesh -> Metrik -> Collider -> USD -> Lint
tools/       Vergleich, Orbit-Views, Manifest, Messungen
LGM/         gepatchte Inferenz (vendored; Upstream-Historie in .git.upstream)
```

## Bauen

Voraussetzung: MSVC-Toolchain, Node für die Tauri-CLI.

```powershell
cd studio
$env:RUSTUP_TOOLCHAIN = "stable-x86_64-pc-windows-msvc"
cargo build
npx --yes @tauri-apps/cli@2 build   # Installer
```

## Dokumentation

- [`ANALYSE_SIMREADY.md`](ANALYSE_SIMREADY.md) — LGM seziert: 415 M Parameter,
  keine 3D-Fusion, und was für SimReady fehlt
- [`RECHERCHE_VIEWS.md`](RECHERCHE_VIEWS.md) — warum mehr Ansichten scheitern
  und welche Modelle es besser können
- [`OPTIMIERUNG_LGM.md`](OPTIMIERUNG_LGM.md) — gemessene Versuche: Schritte,
  OpenVINO, CPU, TripoSplat
- [`studio/README.md`](studio/README.md) — Architektur und der Python-Befund
- [`splatcage/README.md`](splatcage/README.md) — Silhouetten-Carving
- [`SETUP.md`](SETUP.md) — LGM auf Intel-iGPU ohne NVIDIA

Alle Zahlen in diesen Dateien sind auf einem Lunar-Lake-Laptop gemessen
(Core Ultra 7 255U, Arc 64 EU, 11,7 GB shared), nicht geschätzt.
