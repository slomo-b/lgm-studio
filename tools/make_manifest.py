"""Erzeugt studio/manifest.json aus den lokal vorhandenen Modellgewichten.

Zweck: Der Installer soll keine 4,6 GB schleppen. Die App laedt die Dateien
beim ersten Start nach; das Manifest sagt ihr, welche das sind, woher sie
kommen und wie gross sie sind.

Die Pfade spiegeln die Repo-Struktur, damit der Python-Teil die Modelle aus
einem *lokalen Ordner* laden kann (diffusers kann das) statt aus dem
Hugging-Face-Cache.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(
    ROOT, ".hf", "hub",
    "models--ashawkey--imagedream-ipmv-diffusers",
    "snapshots", "73a034178e748421506492e91790cc62d6aefef5",
)
HF_BASE = "https://huggingface.co/ashawkey/imagedream-ipmv-diffusers/resolve/main"
LGM_URL = "https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors"
LGM_FILE = os.path.join(ROOT, "LGM", "pretrained", "model_fp16_fixrot.safetensors")


def main():
    if not os.path.isdir(SNAP):
        sys.exit(f"Snapshot nicht gefunden: {SNAP}")

    files = []
    for dirpath, _, names in os.walk(SNAP):
        for name in names:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, SNAP).replace(os.sep, "/")
            files.append(
                {
                    "id": rel,
                    "path": f"imagedream/{rel}",
                    "url": f"{HF_BASE}/{rel}",
                    "size": os.path.getsize(full),
                }
            )

    if os.path.exists(LGM_FILE):
        files.append(
            {
                "id": "lgm/model_fp16_fixrot",
                "path": "lgm/model_fp16_fixrot.safetensors",
                "url": LGM_URL,
                "size": os.path.getsize(LGM_FILE),
            }
        )

    files.sort(key=lambda f: -f["size"])
    manifest = {
        "version": 1,
        "note": "Modelle fuer LGM Studio. Pfade relativ zum Modellordner.",
        "files": files,
    }

    out = os.path.join(ROOT, "studio", "manifest.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    total = sum(x["size"] for x in files)
    print(f"{out}: {len(files)} Dateien, {total / 1e9:.2f} GB")
    for x in files[:5]:
        print(f"  {x['size'] / 1e6:9.0f} MB  {x['path']}")


if __name__ == "__main__":
    main()
