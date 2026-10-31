"""Konvertiert den ImageDream-UNet nach OpenVINO und misst gegen Torch-XPU.

Vorgehen: erst einen echten Inferenzschritt laufen lassen und die Eingaben
abfangen (die Shapes haengen von CFG, Ansichtszahl und Aufloesung ab -- raten
waere hier teuer), dann mit genau diesen Tensoren tracen, konvertieren und
messen.

    .venv\\Scripts\\python.exe tools\\ov_unet.py --steps 1
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "LGM"))

from core.local_backend import describe, get_device  # noqa: E402
from core.xpu_patches import apply as apply_xpu_patches  # noqa: E402

apply_xpu_patches()

from mvdream.mv_unet import MultiViewUNetModel  # noqa: E402
from mvdream.pipeline_mvdream import MVDreamPipeline  # noqa: E402

MODEL = "ashawkey/imagedream-ipmv-diffusers"

KEYS = ["x", "timesteps", "context", "camera", "ip", "ip_img"]


class UnetWrap(nn.Module):
    """Feste Argumentliste statt kwargs -- so kann getraced werden."""

    def __init__(self, unet, num_frames):
        super().__init__()
        self.unet = unet
        self.num_frames = int(num_frames)

    def forward(self, x, timesteps, context, camera, ip, ip_img):
        return self.unet(
            x=x,
            timesteps=timesteps,
            context=context,
            num_frames=self.num_frames,
            camera=camera,
            ip=ip,
            ip_img=ip_img,
        )


def prepare(path):
    import kiui
    from kiui.op import recenter

    img = kiui.read_image(path, mode="uint8")
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        rgb = img[..., :3].astype(np.float32) * a + (1 - a) * 255.0
        img = np.concatenate([rgb, img[..., 3:4]], axis=-1).astype(np.uint8)
        mask = img[..., 3] > 0
    else:
        mask = np.ones(img.shape[:2], bool)
    img = recenter(img, mask, border_ratio=0.2)
    out = img[..., :3].astype(np.float32) / 255.0
    if img.shape[-1] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        out = out * a + (1 - a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=os.path.join(ROOT, "LGM", "data_test", "catstatue_rgba.png"))
    ap.add_argument("--steps", type=int, default=1, help="nur zum Abfangen der Eingaben")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "ov"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = get_device()
    print(f"[info] torch-Device: {describe(device)}")

    unet = MultiViewUNetModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float16)
    pipe = MVDreamPipeline.from_pretrained(
        MODEL, unet=unet, torch_dtype=torch.float16, trust_remote_code=True
    ).to(device)

    captured = {}

    # Wichtig: die Pipeline ruft `self.unet.forward(**kw)` DIREKT auf. Forward-Hooks
    # von nn.Module greifen dabei nicht -- die Methode muss ersetzt werden.
    orig_forward = pipe.unet.forward

    def patched_forward(**kw):
        if not captured:
            captured.update({k: v for k, v in kw.items() if k in KEYS or k == "num_frames"})
        return orig_forward(**kw)

    pipe.unet.forward = patched_forward
    image = prepare(args.image)
    t0 = time.perf_counter()
    pipe("", image, guidance_scale=5.0, num_inference_steps=args.steps, elevation=0, device=device)
    if device.type == "xpu":
        torch.xpu.synchronize()
    print(f"[capture] {args.steps} Schritt(e) in {time.perf_counter() - t0:.1f}s")
    pipe.unet.forward = orig_forward

    if not captured:
        raise SystemExit("keine Eingaben abgefangen")
    for k, v in captured.items():
        print(f"  {k}: {tuple(v.shape) if isinstance(v, torch.Tensor) else v} {getattr(v, 'dtype', '')}")

    num_frames = captured.get("num_frames", 5)

    # Referenz auf dem XPU
    with torch.no_grad():
        t0 = time.perf_counter()
        ref = pipe.unet(**captured)
        if device.type == "xpu":
            torch.xpu.synchronize()
        t_xpu = time.perf_counter() - t0
    print(f"[torch-xpu] ein UNet-Forward: {t_xpu * 1000:.0f} ms")

    # Tracing auf CPU in fp32 (fp16-Trace ist auf CPU unzuverlaessig)
    wrap = UnetWrap(pipe.unet, num_frames).cpu().float().eval()
    inputs = []
    for k in KEYS:
        t = captured[k]
        inputs.append(t.detach().cpu().float())
    with torch.no_grad():
        traced = torch.jit.trace(wrap, tuple(inputs), strict=False)

    import openvino as ov

    xml_path = os.path.join(args.out, "imagedream_unet.xml")
    core = ov.Core()
    if os.path.exists(xml_path):
        print(f"[ov] vorhandenes IR laden: {xml_path}")
        ov_model = core.read_model(xml_path)
    else:
        ov_model = ov.convert_model(traced, example_input=inputs)
        # fp16-Gewichte: Torch rechnet auch fp16, alles andere waere unfair
        ov.save_model(ov_model, xml_path, compress_to_fp16=True)
        print(f"[ov] konvertiert (fp16-Gewichte): {xml_path}")
    print(f"[ov] {len(ov_model.outputs)} Ausgabe(n), {len(ov_model.get_parameters())} Eingabe(n)")

    cfg = {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_PRECISION_HINT": "f16",
    }

    core = ov.Core()
    results = {}
    for dev in ("GPU", "CPU"):
        try:
            cm = core.compile_model(ov_model, dev, cfg)
            req = cm.create_infer_request()
            np_in = [i.numpy() for i in inputs]
            req.infer(np_in)
            for _ in range(2):
                req.infer(np_in)
            t0 = time.perf_counter()
            for _ in range(3):
                req.infer(np_in)
            dt = (time.perf_counter() - t0) / 3
            out = np.array(req.get_output_tensor(0).data)
            results[dev] = out
            print(f"[{dev}] {dt * 1000:.0f} ms  (Speedup gegen Torch-XPU: {t_xpu / dt:.1f}x)")
        except Exception as e:
            print(f"[{dev}] FEHLER: {e}")

    ref_np = ref.detach().cpu().float().numpy()
    for dev, out in results.items():
        diff = float(np.abs(out.reshape(ref_np.shape) - ref_np).max())
        rel = float(np.abs(out.reshape(ref_np.shape) - ref_np).mean() / (np.abs(ref_np).mean() + 1e-9))
        print(f"[{dev}] max|diff| {diff:.4f}  mittlere rel. Abweichung {rel * 100:.2f}%")


if __name__ == "__main__":
    main()
