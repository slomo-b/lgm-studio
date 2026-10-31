"""Lokale Machbarkeit: was kostet ein Generator mit mehr Views auf der iGPU?

Misst den Forward des LGM-Generators (415 M) mit 4 und 8 Ansichten bei 256^2
und rechnet daraus hoch, was eine echte Generierung lokal kosten wuerde.

Hintergrund: SEVA (1,3 B, 576p, flash-attn) ist auf dieser Hardware chancenlos.
Die Frage ist, ob ein *kleines* Modell mit 8 Views lokal in Sekunden laufen
kann -- oder ob auch das aussichtslos ist.

    .venv\\Scripts\\python.exe bench_local_views.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LGM"))

from core.local_backend import get_device, describe  # noqa: E402
from core.options import config_defaults  # noqa: E402
from core.xpu_patches import apply as apply_xpu_patches  # noqa: E402

apply_xpu_patches()

from core.models import LGM  # noqa: E402


def sync(device):
    if device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / runs


def main():
    device = get_device()
    print(f"[info] {describe(device)}")

    opt = config_defaults["big"]
    opt.lambda_lpips = 0.0
    model = LGM(opt).to(device=device, dtype=torch.float16).eval()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[info] LGM big: {n_par:.0f} M Parameter, splat_size={opt.splat_size}")

    # Kuenstlicher Ray-Block: 6 Kanaele, damit die 9 Eingabekanaele stimmen
    def make_input(views):
        img = torch.randn(1, views, 9, opt.input_size, opt.input_size, device=device, dtype=torch.float16)
        return img

    print()
    print(f"{'views':>6} {'forward':>10} {'  x 20 Schritte':>16} {'  x 30 Schritte':>16}")
    print("-" * 52)
    base = {}
    for v in (4, 8, 12):
        x = make_input(v)
        with torch.no_grad():
            def run(x=x):
                return model.forward_gaussians(x)

            try:
                t = timeit(run, device)
            except Exception as e:
                print(f"{v:>6}  FEHLER: {e}")
                continue
        base[v] = t
        print(
            f"{v:>6} {t*1000:>9.0f}ms {t*20:>15.1f}s {t*30:>15.1f}s"
        )

    # Vergleich: heutiger Stand (30 Diffusionsschritte + 1x Forward, 4 Views)
    print()
    if 4 in base:
        t4 = base[4]
        print(f"[heute] 30 Diffusionsschritte @4 Views + 1x Generator = {t4*31:.0f} s (grob)")
    if 8 in base:
        t8 = base[8]
        print(f"[8 Views] ein Forward = {t8:.2f} s; mit 12 Schritten waeren es {t8*12:.0f} s")
        print(f"[8 Views] Ziel < 10 s pro Asset braucht < {10/12*1000:.0f} ms pro Schritt")


if __name__ == "__main__":
    main()
