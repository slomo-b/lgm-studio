"""Benchmark: MVDream-UNet mit und ohne XPU-Graph-Capture.

Der UNet ist der Engpass (30 Schritte pro Bild). Auf der iGPU sind das
~2,8 s pro Schritt fuer ein winziges 32x32-Latent -- das riecht nach
Kernel-Launch-Overhead, nicht nach Rechenzeit. XPU-Graphs nehmen den
kompletten Forward in einen einzigen Launch.

Die Eingabe-Shapes werden nicht geraten, sondern aus einem echten
Pipeline-Schritt mitgeschrieben.

    ..\\.venv\\Scripts\\python.exe ..\\bench_unet.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

LGM = Path(__file__).resolve().parent / "LGM"
sys.path.insert(0, str(LGM))
os.environ.setdefault("HF_HOME", str(LGM.parent / ".hf"))

from core.xpu_patches import apply  # noqa: E402

apply()

from mvdream.mv_unet import MultiViewUNetModel  # noqa: E402
from mvdream.pipeline_mvdream import MVDreamPipeline  # noqa: E402

REPO = "ashawkey/imagedream-ipmv-diffusers"
DT = torch.bfloat16 if os.environ.get("DT") == "bf16" else torch.float16


def bench(fn, iters=3, warmup=1):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / iters


class Wrapper(torch.nn.Module):
    """Positionale Argumente, weil make_graphed_callables das verlangt."""

    def __init__(self, unet, num_frames):
        super().__init__()
        self.unet = unet
        self.num_frames = num_frames

    def forward(self, x, t, ctx, cam, ip, ip_img):
        out = self.unet(x, t, context=ctx, num_frames=self.num_frames,
                        camera=cam, ip=ip, ip_img=ip_img)
        return out


def main():
    print("lade UNet ...", flush=True)
    unet = MultiViewUNetModel.from_pretrained(REPO, subfolder="unet", torch_dtype=DT).to("xpu")
    unet.eval()

    print("lade Pipeline ...", flush=True)
    pipe = MVDreamPipeline.from_pretrained(REPO, unet=unet, torch_dtype=DT,
                                           trust_remote_code=True).to("xpu")

    recorded = {}
    orig_forward = unet.forward

    def spy(*a, **kw):
        if not recorded:
            recorded["args"] = a
            recorded["kwargs"] = {k: (v.shape if torch.is_tensor(v) else v) for k, v in kw.items()}
            recorded["tensors"] = {k: v for k, v in kw.items() if torch.is_tensor(v)}
        return orig_forward(*a, **kw)

    unet.forward = spy
    print("ein echter Pipeline-Schritt (1 Schritt) ...", flush=True)
    img = np.full((256, 256, 3), 0.5, dtype=np.float32)
    with torch.no_grad():
        pipe('', img, guidance_scale=5.0, num_inference_steps=1, elevation=0)
    unet.forward = orig_forward

    shapes = {k: tuple(v.shape) for k, v in recorded["tensors"].items()}
    print("  aufgezeichnet:", shapes, flush=True)

    kw = recorded["tensors"]
    num_frames = recorded["kwargs"]["num_frames"]
    args = (kw["x"], kw["timesteps"], kw["context"], kw["camera"], kw["ip"], kw["ip_img"])

    w = Wrapper(unet, num_frames).eval()
    for p in w.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        t0 = time.perf_counter()
        w(*args)
        torch.xpu.synchronize()
        print(f"  kalt:   {time.perf_counter() - t0:6.2f}s", flush=True)

        base = bench(lambda: w(*args), iters=3)
    print(f"  ohne Graph: {base * 1000:7.0f} ms  ({1 / base:.2f} fps)", flush=True)

    try:
        g = torch.xpu.make_graphed_callables(w, args)
        with torch.no_grad():
            t0 = time.perf_counter()
            g(*args)
            torch.xpu.synchronize()
            cold = time.perf_counter() - t0
            dt = bench(lambda: g(*args), iters=5, warmup=1)
        print(f"  mit Graph:  capture {cold:5.1f}s, danach {dt * 1000:7.0f} ms "
              f"({1 / dt:.2f} fps)  -> {base / dt:.1f}x schneller", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"  Graph-Capture FEHLER {type(exc).__name__}: {str(exc)[:300]}", flush=True)


if __name__ == "__main__":
    main()
