"""Wendet den transformers-Patch fuer Intel-XPU an (idempotent).

Hintergrund: `CLIPTextTransformer.forward` holt den EOS-Token per
advanced indexing (`last_hidden_state[arange, argmax]`). Das ist ein
Index-Kernel, und Index-Kernel haengen auf dieser Plattform reproduzierbar
(torch 2.8 wie 2.15). Der Patch ersetzt die Stelle durch eine gather-freie
Variante (Vergleichsmaske + gewichtete Summe), numerisch identisch.

Nach einem Neuaufbau der venv einmal ausfuehren:

    .venv\\Scripts\\python.exe patch_deps.py
"""

import sys
from pathlib import Path

MARKER = "def _select_tokens("

HELPER = '''logger = logging.get_logger(__name__)


def _select_tokens(hidden: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """[LOCAL PATCH XPU] gather-free replacement for ``hidden[arange, sel]``.

    Advanced indexing uses an index kernel that hangs on Intel XPU (tested with
    torch 2.8 and 2.15). A one-hot mask via comparison + weighted sum is
    numerically identical.
    """
    one_hot = (torch.arange(hidden.shape[1], device=hidden.device) == sel.unsqueeze(-1)).to(hidden.dtype)
    return (hidden * one_hot.unsqueeze(-1)).sum(dim=1)'''

OLD_A = '''            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]'''

NEW_A = '''            # [LOCAL PATCH XPU] advanced indexing (= index kernel) hangs on Intel XPU.
            # Gather-free equivalent: one-hot via comparison + weighted sum.
            _sel = input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1)
            pooled_output = _select_tokens(last_hidden_state, _sel)'''

OLD_B = '''            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]'''

NEW_B = '''            _sel = (
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1)
            )
            pooled_output = _select_tokens(last_hidden_state, _sel)'''


def main() -> int:
    try:
        import transformers
    except ImportError:
        print("transformers ist nicht installiert.")
        return 1

    path = Path(transformers.__file__).parent / "models" / "clip" / "modeling_clip.py"
    if not path.is_file():
        print(f"nicht gefunden: {path}")
        return 1

    src = path.read_text(encoding="utf-8")

    if MARKER in src:
        print(f"bereits gepatcht: {path}")
        return 0

    for old in (OLD_A, OLD_B):
        if old not in src:
            print("unerwarteter Dateiinhalt - transformers-Version passt nicht (erwartet 4.40.2).")
            return 2

    src = src.replace("logger = logging.get_logger(__name__)", HELPER, 1)
    src = src.replace(OLD_A, NEW_A, 1)
    src = src.replace(OLD_B, NEW_B, 1)
    path.write_text(src, encoding="utf-8")

    print(f"gepatcht: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
