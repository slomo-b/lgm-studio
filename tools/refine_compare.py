"""Vorher/Nachher des Feinschliffs auf den vier LGM-Ansichten."""

import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kiui.cam import orbit_camera  # noqa: E402

from simready import diffrender as DR  # noqa: E402
from simready import splat as S  # noqa: E402


def cam(az):
    pose = orbit_camera(0.0, az, radius=1.5, opengl=True)
    eye = torch.tensor(pose[:3, 3], dtype=torch.float32)
    right = torch.tensor(pose[:3, 0], dtype=torch.float32)
    up = torch.tensor(pose[:3, 1], dtype=torch.float32)
    back = torch.tensor(pose[:3, 2], dtype=torch.float32)
    return torch.stack([right, -up, -back], dim=0), eye


def render(path, dev, size=256):
    g = S.clean(S.load_gaussians(path))
    args = [
        torch.tensor(g[k], dtype=torch.float32, device=dev)
        for k in ("xyz", "scale", "rot", "opacity", "color")
    ]
    out = []
    for i in range(4):
        R, eye = cam(90.0 * i)
        with torch.no_grad():
            img, _ = DR.render(*args, R.to(dev), eye.to(dev), size=size)
        out.append((img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "catstatue_rgba"
    dev = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    before = render(f"LGM/workspace_test/{name}.ply", dev)
    after = render(f"out/refine/{name}_refined.ply", dev)
    sheet = np.concatenate([np.concatenate(before, axis=1), np.concatenate(after, axis=1)], axis=0)
    out = f"out/refine/{name}_vergleich.png"
    Image.fromarray(sheet).save(out)
    print(f"[out] {out}  (oben vorher, unten nachher)")


if __name__ == "__main__":
    main()
