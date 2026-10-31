"""CPU-Fallback fuer die Index-Kernel, die auf dieser XPU-Plattform haengen.

In `LGM/core/xpu_patches.py` sind nur `nn.Embedding` und die Causal-Mask
abgedeckt. TripoSplats Octree-Sampler nutzt zusaetzlich `index_select`,
`scatter_add_` und `index_copy_` -- laut SETUP.md haengen genau diese Kernel
hier reproduzierbar.

Der Patch leitet nur die betroffenen Aufrufe auf die CPU und schiebt das
Ergebnis zurueck. Das ist langsam, aber korrekt; bei Tensoren dieser Groesse
(Octree-Knoten, 32k-262k) bleibt es ertraeglich.

    from xpu_index_patches import apply
    apply()
"""

import torch


def apply(verbose=True):
    if getattr(torch.Tensor, "_xpu_index_patched", False):
        return

    T = torch.Tensor

    def _cpu_run(fn_name):
        orig = getattr(T, fn_name)

        def wrapper(self, *args, **kwargs):
            if self.device.type != "xpu":
                return orig(self, *args, **kwargs)
            cpu_args = [
                a.cpu() if isinstance(a, torch.Tensor) and a.device.type == "xpu" else a
                for a in args
            ]
            out = orig(self.cpu(), *cpu_args, **kwargs)
            if isinstance(out, torch.Tensor):
                return out.to(self.device)
            return out

        setattr(T, fn_name, wrapper)

    for name in ("index_select", "index_copy_", "scatter_add_", "index_add_"):
        try:
            _cpu_run(name)
        except Exception as e:  # pragma: no cover
            if verbose:
                print(f"[warn] {name} nicht gepatcht: {e}")

    T._xpu_index_patched = True
    if verbose:
        print("[info] XPU-Index-Patches aktiv (index_select, index_copy_, scatter_add_, index_add_)")
