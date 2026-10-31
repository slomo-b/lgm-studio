"""[LOCAL PATCH] LGM als wiederverwendbare Klasse statt Skript.

Damit koennen Server und CLI dieselbe Pipeline nutzen und die Modelle
zwischen Auftraegen im Speicher bleiben -- das spart pro Bild ~26 s
(11 s Laden + 13 s Kernel-Kompilierung des ersten Schritts).
"""

import os
import time

import imageio
import kiui
import numpy as np
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from kiui.op import recenter
from safetensors.torch import load_file

from core.local_backend import describe, get_autocast_dtype, get_device, model_dtype
from core.xpu_patches import apply as _apply_xpu_patches

# Muss vor dem ersten Modelllauf passieren: ohne diese Patches haengen
# Index-Kernel auf der iGPU (Causal-Mask, nn.Embedding).
_apply_xpu_patches()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
REPO = "ashawkey/imagedream-ipmv-diffusers"


def model_repo():
    """Lokalen Modellordner bevorzugen, sonst Hugging Face.

    Die App laedt die Gewichte beim ersten Start in <projekt>/models. Wenn sie
    dort liegen, wird nichts mehr aus dem Netz geholt -- das ist der
    Offline-Betrieb und gleichzeitig die Voraussetzung fuer einen kleinen
    Installer.
    """
    base = os.environ.get("LGM_MODEL_DIR")
    if not base:
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base = os.path.join(here, "models")
    local = os.path.join(base, "imagedream")
    if os.path.exists(os.path.join(local, "model_index.json")):
        return local
    return REPO
GUIDANCE = float(os.environ.get("LGM_GUIDANCE", "6.0"))
# Sampler: "ddim" (upstream) oder "dpm" (DPM-Solver++ mit Karras-Sigmas)
SCHEDULER = os.environ.get("LGM_SCHEDULER", "ddim").lower()

# Stages, die der Server fuer die Fortschrittsanzeige nutzt
STAGE_LOAD, STAGE_VIEWS, STAGE_GAUSS, STAGE_WRITE = 0, 1, 2, 3


class LgmPipeline:
    """Haelt LGM + ImageDream geladen und erzeugt daraus .ply-Dateien."""

    def __init__(self, opt, checkpoint, device=None, on_progress=None, on_profile=None):
        self.opt = opt
        self.device = device or get_device()
        self._progress = on_progress or (lambda *a, **k: None)
        self._profile = on_profile or (lambda *a, **k: None)

        self.model = None
        self.pipe = None
        self.rays = None
        self.bg_remover = None

        self._load_lgm(checkpoint)
        self._load_rembg()

    # ------------------------------------------------------------ Aufbau

    def _load_lgm(self, checkpoint):
        from core.models import LGM

        t0 = time.perf_counter()
        self._progress(STAGE_LOAD, 0.02, "job.loading")
        model = LGM(self.opt)

        if checkpoint:
            if checkpoint.endswith("safetensors"):
                ckpt = load_file(checkpoint, device="cpu")
            else:
                ckpt = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(ckpt, strict=False)
            print(f"[INFO] Loaded checkpoint from {checkpoint}", flush=True)

        model = model.to(device=self.device, dtype=model_dtype(self.device))
        model.eval()
        self.model = model
        self._profile("lgm_load", t0)
        print(f"[INFO] backend: {describe(self.device)}", flush=True)

        self.rays = model.prepare_default_rays(self.device)

    def _load_rembg(self):
        t0 = time.perf_counter()
        self.bg_remover = rembg.new_session()
        self._profile("rembg_init", t0)

    def _ensure_pipe(self):
        """ImageDream nur laden, wenn wirklich gebraucht (~6 s, 5 GB)."""
        if self.pipe is not None:
            return
        from mvdream.mv_unet import MultiViewUNetModel
        from mvdream.pipeline_mvdream import MVDreamPipeline

        t0 = time.perf_counter()
        self._progress(STAGE_LOAD, 0.05, "job.loading")
        # UNet aus dem lokalen, SDPA-gepatchten Modul -- das HF-Repo bringt eine
        # eigene Kopie mit, die hart auf xformers importiert.
        unet = MultiViewUNetModel.from_pretrained(
            model_repo(), subfolder="unet", torch_dtype=torch.float16
        )
        pipe = MVDreamPipeline.from_pretrained(
            model_repo(), unet=unet, torch_dtype=torch.float16, trust_remote_code=True,
        )
        if SCHEDULER == "dpm":
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True,
            )
            print("[INFO] scheduler: DPM-Solver++ (Karras)", flush=True)

        self.pipe = pipe
        self._profile("pipe_load", t0)

    def set_progress(self, callback):
        """Fortschritts-Callback austauschen (Server setzt ihn pro Auftrag)."""
        self._progress = callback or (lambda *a, **k: None)

    def warmup(self, steps=2):
        """Kernel einmal kompilieren, damit der erste echte Auftrag schnell ist."""
        img = np.full((256, 256, 3), 0.5, dtype=np.float32)
        self.generate_from_array(img, workspace=None, steps=steps, dump_views=False)
        self.pipe.to("cpu")
        self._empty_cache()

    # ------------------------------------------------------------ Lauf

    def _empty_cache(self):
        if self.device.type == "xpu":
            torch.xpu.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def generate(self, image_path, workspace=None, steps=16, seed=None,
                 dump_views=True, name=None):
        image = kiui.read_image(image_path, mode="uint8")
        return self.generate_from_array(
            image, workspace=workspace, steps=steps, seed=seed,
            name=name or os.path.splitext(os.path.basename(image_path))[0],
            dump_views=dump_views,
        )

    def generate_from_array(self, image, workspace=None, steps=16, seed=None,
                            name="image", dump_views=True):
        """Bild (uint8 RGB/RGBA) -> dict mit Pfad zur .ply und Zeitmessung."""
        self._ensure_pipe()
        timing = {}
        t_all = time.perf_counter()

        # --- Hintergrund entfernen
        self._progress(STAGE_LOAD, 0.10, "job.bg")
        t0 = time.perf_counter()
        carved = rembg.remove(image, session=self.bg_remover)
        mask = carved[..., -1] > 0
        cut = recenter(carved, mask, border_ratio=0.2).astype(np.float32) / 255.0
        if cut.shape[-1] == 4:
            cut = cut[..., :3] * cut[..., 3:4] + (1 - cut[..., 3:4])
        timing["rembg"] = time.perf_counter() - t0
        self._profile("rembg", t0)

        # --- Multi-View-Diffusion
        self.pipe.to(self.device)
        gen = torch.Generator(device="cpu").manual_seed(int(seed)) if seed is not None else None

        state = {"last": time.perf_counter()}

        def step_cb(i, _t, _latents):
            now = time.perf_counter()
            state.setdefault("steps", []).append(round(now - state["last"], 3))
            state["last"] = now
            self._progress(STAGE_VIEWS, 0.12 + 0.66 * (i + 1) / steps, "job.views", {"n": i + 1})

        t0 = time.perf_counter()
        mv = self.pipe("", cut, guidance_scale=GUIDANCE, num_inference_steps=steps,
                       elevation=0, generator=gen, callback=step_cb)
        timing["diffusion"] = time.perf_counter() - t0
        self._profile("diffusion", t0)

        mv = np.stack([mv[1], mv[2], mv[3], mv[0]], axis=0)   # [4,256,256,3]

        if dump_views and workspace:
            try:
                mv_dir = os.path.join(workspace, "mv")
                os.makedirs(mv_dir, exist_ok=True)
                for i, im in enumerate(mv):
                    imageio.imwrite(os.path.join(mv_dir, f"{name}_view{i}.png"),
                                    (im * 255).astype(np.uint8))
            except Exception as exc:  # noqa: BLE001
                print("[WARN] multiview dump failed:", exc, flush=True)

        # --- Platz fuer den LGM-Decoder machen
        t0 = time.perf_counter()
        self.pipe.to("cpu")
        self._empty_cache()
        timing["offload"] = time.perf_counter() - t0
        self._profile("offload", t0)

        # --- Gaussians
        self._progress(STAGE_GAUSS, 0.82, "job.gauss")
        t0 = time.perf_counter()
        inp = torch.from_numpy(mv).permute(0, 3, 1, 2).float().to(self.device)
        inp = F.interpolate(inp, size=(self.opt.input_size, self.opt.input_size),
                            mode="bilinear", align_corners=False)
        inp = TF.normalize(inp, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
        inp = torch.cat([inp, self.rays], dim=1).unsqueeze(0)

        with torch.no_grad():
            with torch.autocast(device_type=self.device.type,
                                dtype=get_autocast_dtype(self.device)):
                gaussians = self.model.forward_gaussians(inp)
            if self.device.type == "xpu":
                torch.xpu.synchronize()
        timing["lgm_decode"] = time.perf_counter() - t0
        self._profile("lgm_decode", t0)

        # --- schreiben
        self._progress(STAGE_WRITE, 0.94, "job.write")
        ply_path = None
        if workspace:
            os.makedirs(workspace, exist_ok=True)
            ply_path = os.path.join(workspace, name + ".ply")
            t0 = time.perf_counter()
            self.model.gs.save_ply(gaussians, ply_path)
            timing["save_ply"] = time.perf_counter() - t0
            self._profile("save_ply", t0)

        timing["total"] = time.perf_counter() - t_all
        timing["steps"] = state.get("steps", [])
        return {"name": name, "ply": ply_path, "timing": timing}

    # ------------------------------------------------------------ Aufraeumen

    def unload_pipe(self):
        """ImageDream freigeben, wenn der Speicher anderswo gebraucht wird."""
        if self.pipe is not None:
            self.pipe.to("cpu")
            self.pipe = None
            self._empty_cache()

    def stats(self):
        return {
            "device": describe(self.device),
            "lgm_loaded": self.model is not None,
            "pipe_loaded": self.pipe is not None,
        }
