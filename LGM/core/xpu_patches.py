"""[LOCAL PATCH] Workarounds fuer PyTorch-XPU auf Lunar Lake / Windows.

Problem: Index-Kernel (index_select / index_put_ / damit auch masked_fill_ und
nn.Embedding) haengen auf dieser Plattform komplett. Getestet mit torch 2.8 und
2.15, jeweils reproduzierbar; Konvolutionen und SDPA laufen dagegen normal.
Betroffen sind hier nur die CLIP-Text-/Image-Encoder (77 Tokens) und die
Attention-Maske (77x77) -- beides rechnen wir auf der CPU, das kostet nichts.
"""

import torch


def _patch_causal_mask():
    try:
        from transformers.modeling_attn_mask_utils import AttentionMaskConverter
    except ImportError:
        return

    if getattr(AttentionMaskConverter, "_lgm_xpu_patched", False):
        return

    orig = AttentionMaskConverter._make_causal_mask

    def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0, sliding_window=None):
        device = torch.device(device)
        if device.type != "xpu":
            return orig(input_ids_shape, dtype, device, past_key_values_length, sliding_window)
        mask = orig(input_ids_shape, dtype, torch.device("cpu"), past_key_values_length, sliding_window)
        return mask.to(device)

    AttentionMaskConverter._make_causal_mask = staticmethod(_make_causal_mask)
    AttentionMaskConverter._lgm_xpu_patched = True


def _patch_embedding():
    import torch.nn as nn

    if getattr(nn.Embedding, "_lgm_xpu_patched", False):
        return

    orig = nn.Embedding.forward

    def forward(self, input):
        if input.device.type != "xpu":
            return orig(self, input)
        # Lookup auf der CPU (paar hundert MB, einmalig), Ergebnis zurueck aufs Geraet
        w = self.weight.detach().float().cpu()
        out = torch.nn.functional.embedding(
            input.cpu(),
            w,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        return out.to(device=input.device, dtype=self.weight.dtype)

    nn.Embedding.forward = forward
    nn.Embedding._lgm_xpu_patched = True


def apply():
    _patch_causal_mask()
    _patch_embedding()

