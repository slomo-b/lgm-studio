"""[LOCAL PATCH] Compatibility shim.

`ashawkey/imagedream-ipmv-diffusers` declares `"unet": ["mv_unet",
"MultiViewUNetModel"]` in its model_index.json, and its remote pipeline does
`from mv_unet import MultiViewUNetModel, get_camera`. Upstream LGM only ships
`mvdream/mv_unet.py`, which is why the import works there by accident of the
working directory. This file makes `import mv_unet` resolve to the *patched*
local module (SDPA instead of xformers) no matter who asks.
"""

from mvdream.mv_unet import *  # noqa: F401,F403
from mvdream.mv_unet import MultiViewUNetModel, get_camera, memory_efficient_attention  # noqa: F401
