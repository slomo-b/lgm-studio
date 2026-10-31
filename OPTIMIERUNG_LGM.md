# Gemessene Optimierungsversuche für LGM

Damit dieselben Versuche nicht zweimal gemacht werden. Alle Zahlen auf dem
Lunar-Lake-Laptop (Core Ultra 7 255U, Arc 64 EU, 11,7 GB shared, 32 GB RAM).

---

## 1. Diffusionsschritte

Gemessen mit `bench_steps.py`, gleicher Seed, 4 Ansichten:

| Schritte | Ansichten identisch zu 30? | Gaussians | Oberfläche |
|---|---|---|---|
| 30 | Referenz | 33 041 | glatt |
| 24 | ja | — | glatt |
| 20 | ja | 33 879 | glatt |
| **16** | **ja** | **33 279 (−1,8 %)** | **glatt** |
| 12 | ja (2D), **nein** (3D) | 31 371 (−7,4 %) | körnig, raue Konturen |
| 8 | nein | — | kaputt |

**Ergebnis: 16 Schritte sind der Sweet Spot.** Visuell gleichwertig zu 20,
aber 20 % weniger Diffusionszeit. Unter 16 leidet sichtbar die Geometrie.

Praktisch: `LGM_STEPS=16`. Stand default ist 20.

---

## 2. OpenVINO auf der iGPU — kein Gewinn

Der UNet wurde per `torch.jit.trace` konvertiert (`tools/ov_unet.py`), echte
Eingaben aus einem Inferenzschritt abgefangen:

```
x (10,4,32,32)  timesteps (10,)  context (10,77,1024)
camera (10,16)  ip (10,257,1280) ip_img (2,4,32,32)   num_frames 5
```

| Pfad | Zeit pro UNet-Forward | Abweichung zu Torch |
|---|---|---|
| Torch-XPU fp16 | **2 236–2 404 ms** | Referenz |
| OpenVINO GPU, fp32-IR | 3 345 ms (0,7×) | 2,59 % |
| OpenVINO GPU, fp16-IR + `INFERENCE_PRECISION_HINT=f16` | **2 347 ms (1,0×)** | 2,39 % |
| OpenVINO CPU | 11 590–12 546 ms (0,2×) | 0,19 % |

**Ergebnis: Parität, kein Speedup.** Mit fp16-Gewichten holt OpenVINO exakt
auf, mehr nicht — und rechnet dabei ungenauer als Torch. Der Grund ist die
Größe: 10 Latents à 32×32 sind viele kleine Operationen, da frisst der
Graph-Overhead den Rechenvorteil auf.

Rohleistung wäre da — ein 3×3-Conv-Benchmark zeigt **1 171 GFLOPS auf der GPU
gegen 239 auf der CPU** (4,9×). Es kommt nur nicht an.

---

## 3. TripoSplat — läuft, aber unbrauchbar

Siehe `RECHERCHE_VIEWS.md` §8. Kurz:

| | XPU | CPU |
|---|---|---|
| Modell laden | 14,4 s | 8,6 s |
| Schritt 1 | 539 s (Kompilierung) | nach 5,5 min nicht fertig |
| Beharrung | ~110 s/Schritt | nie erreicht |
| 20 Schritte | **~40 min** | — |

Auf der CPU stand der Prozess bei **3 % Last** — der fest verdrahtete
bfloat16-Pfad findet dort keinen schnellen Kernel. Quantisierung hilft nicht
(2–4×, gebraucht würden 25×), und die HF-Release enthält ohnehin nur fp16.

---

## 4. Was übrig bleibt

| Hebel | Gewinn | Aufwand |
|---|---|---|
| `LGM_STEPS=16` | **−20 % Diffusionszeit** | eine Zeile |
| OpenVINO | 0 % | erledigt, verworfen |
| CPU statt iGPU | −80 % | erledigt, verworfen |
| andere Modelle lokal | unbrauchbar (40 min) | erledigt, verworfen |
| gemietete GPU | **68 s → ~10 s** | 5–20 ct pro Lauf |

Auf dieser Hardware ist LGM bei ~95 s pro Asset am Limit. Der nächste echte
Sprung ist Hardware, nicht Software.
