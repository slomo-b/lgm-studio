"""Welche Tensor-Operationen funktionieren auf dieser XPU wirklich?

Hintergrund: SETUP.md dokumentiert, dass Index-Kernel (index_select, index_put_,
advanced indexing) hier haengen. Ein differenzierbarer Splat-Renderer braucht
genau solche Operationen (Sortieren + Streuen). Dieses Skript prueft der Reihe
nach, was geht -- und haengt es bei einer Operation, ist das die Antwort.

Jede Zeile wird vor der Ausfuehrung geflusht, damit im Timeout-Fall sichtbar
ist, welche Operation schuld war.

    .venv\\Scripts\\python.exe tools\\xpu_opcheck.py
"""

import sys
import time

import torch

DEV = "xpu" if torch.xpu.is_available() else "cpu"
N, P = 20000, 4096


def step(name, fn):
    print(f"[..] {name}", end="", flush=True)
    t0 = time.perf_counter()
    fn()
    if DEV == "xpu":
        torch.xpu.synchronize()
    print(f"\r[ok] {name:34s} {(time.perf_counter() - t0) * 1000:8.1f} ms", flush=True)


def main():
    print(f"Device: {DEV}", flush=True)
    torch.manual_seed(0)

    x = torch.rand(N, device=DEV)
    idx = torch.randint(0, P, (N,), device=DEV)
    mat = torch.rand(P, 8, device=DEV)
    pix = torch.randint(0, P, (N,), device=DEV)
    vals = torch.rand(N, 3, device=DEV)

    step("sort", lambda: torch.sort(x))
    step("argsort", lambda: torch.argsort(x))
    step("cumsum", lambda: torch.cumsum(x, 0))
    step("cumprod", lambda: torch.cumprod(1.0 - x * 1e-3, 0))
    step("gather (dim0)", lambda: torch.gather(mat, 0, idx.unsqueeze(1).expand(N, 8)))
    step("scatter_add", lambda: torch.zeros(P, 3, device=DEV).scatter_add_(0, pix.unsqueeze(1).expand(N, 3), vals))
    step("index_select", lambda: mat.index_select(0, idx))
    step("index_add_", lambda: torch.zeros(P, 3, device=DEV).index_add_(0, pix, vals))
    step("masked_select", lambda: x[x > 0.5])
    step("advanced index x[idx]", lambda: mat[idx])
    step("unique", lambda: torch.unique(idx))

    # Zwei Ableitungen: das entscheidet ueber Training auf XPU
    print("--- Autograd ---", flush=True)
    w = torch.rand(N, 3, device=DEV, requires_grad=True)
    try:
        out = torch.zeros(P, 3, device=DEV).scatter_add_(0, pix.unsqueeze(1).expand(N, 3), w).sum()
        out.backward()
        print(f"[ok] backward durch scatter_add          grad-mean {w.grad.mean():.4f}", flush=True)
    except Exception as e:
        print(f"[--] backward scatter_add: {e}", flush=True)

    print("\nfertig", flush=True)


if __name__ == "__main__":
    main()
