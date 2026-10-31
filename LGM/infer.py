"""LGM-Inferenz: Bild -> Gaussian-Splat-.ply.

Duenner CLI-Wrapper um `core/pipeline.LgmPipeline`. Der Server benutzt
dieselbe Klasse direkt und haelt die Modelle im Speicher.

    python infer.py big --resume pretrained/model_fp16_fixrot.safetensors \
        --workspace workspace_test --test_path data_test

Umgebungsvariablen:
    LGM_STEPS=16     Diffusionsschritte (12 schnell / 16 Standard / 30 fein)
    LGM_SEED=42      reproduzierbares Ergebnis
    LGM_PROFILE=1    Zeitmessung pro Phase nach stderr

Hinweis: das 360-Grad-Vorschauvideo von upstream fehlt hier -- es braucht den
CUDA-Rasterizer (`diff_gaussian_rasterization`). Ersatz: `render_preview.py`
oder direkt der WebGL-Viewer der App.
"""

import glob
import json
import os
import time

import tyro

from core.options import AllConfigs
from core.xpu_patches import apply as apply_xpu_patches

apply_xpu_patches()

from core.pipeline import LgmPipeline  # noqa: E402

PROFILE = os.environ.get("LGM_PROFILE") == "1"
NUM_STEPS = int(os.environ.get("LGM_STEPS", "16"))
SEED = os.environ.get("LGM_SEED")
_PROF = {}


def on_profile(name, t0):
    if not PROFILE:
        return
    dt = round(time.perf_counter() - t0, 3)
    _PROF[name] = round(_PROF.get(name, 0.0) + dt, 3)
    print(f"[PROF] {name:<12s} {dt:7.2f}s", flush=True)


def on_progress(stage, fraction, key, params=None):
    if PROFILE:
        print(f"[PROF] stage {stage} {fraction * 100:5.1f}%  {key}", flush=True)


def main():
    opt = tyro.cli(AllConfigs)

    if opt.test_path is None:
        raise SystemExit("--test_path fehlt")

    pipeline = LgmPipeline(opt, opt.resume, on_progress=on_progress, on_profile=on_profile)

    paths = (glob.glob(os.path.join(opt.test_path, "*"))
             if os.path.isdir(opt.test_path) else [opt.test_path])

    for path in paths:
        print(f"[INFO] Processing {path}", flush=True)
        res = pipeline.generate(path, workspace=opt.workspace, steps=NUM_STEPS, seed=SEED)
        t = res["timing"]
        print(f"[INFO] fertig in {t['total']:.1f}s -> {res['ply']}", flush=True)
        if PROFILE:
            steps = t.get("steps", [])
            if steps:
                s = sorted(steps)
                _PROF["steps_n"] = len(s)
                _PROF["step_median"] = s[len(s) // 2]
            _PROF["steps"] = steps

    if PROFILE:
        print("[PROF_JSON] " + json.dumps(_PROF), flush=True)


if __name__ == "__main__":
    main()
