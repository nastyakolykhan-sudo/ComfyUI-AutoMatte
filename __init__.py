import logging

from .nodes import NODE_CLASS_MAPPINGS as _CORE, NODE_DISPLAY_NAME_MAPPINGS as _CORE_NAMES
from .layers import NODE_CLASS_MAPPINGS as _LAYERS, NODE_DISPLAY_NAME_MAPPINGS as _LAYER_NAMES
from .effects import NODE_CLASS_MAPPINGS as _FX, NODE_DISPLAY_NAME_MAPPINGS as _FX_NAMES
from .finish import NODE_CLASS_MAPPINGS as _FIN, NODE_DISPLAY_NAME_MAPPINGS as _FIN_NAMES
from .retime import NODE_CLASS_MAPPINGS as _RT, NODE_DISPLAY_NAME_MAPPINGS as _RT_NAMES
from .relight import NODE_CLASS_MAPPINGS as _RL, NODE_DISPLAY_NAME_MAPPINGS as _RL_NAMES
from .passes import NODE_CLASS_MAPPINGS as _PS, NODE_DISPLAY_NAME_MAPPINGS as _PS_NAMES

NODE_CLASS_MAPPINGS = {**_CORE, **_LAYERS, **_FX, **_FIN, **_RT, **_RL, **_PS}
NODE_DISPLAY_NAME_MAPPINGS = {**_CORE_NAMES, **_LAYER_NAMES, **_FX_NAMES, **_FIN_NAMES, **_RT_NAMES, **_RL_NAMES, **_PS_NAMES}

# Text on Surface needs OpenCV (opencv-python-headless, in requirements.txt): without it the other nodes still load.
try:
    from .text_surface import NODE_CLASS_MAPPINGS as _TX, NODE_DISPLAY_NAME_MAPPINGS as _TX_NAMES
    NODE_CLASS_MAPPINGS.update(_TX)
    NODE_DISPLAY_NAME_MAPPINGS.update(_TX_NAMES)
except ImportError as e:  # pragma: no cover
    logging.warning("ComfyUI-AutoMatte: Text on Surface not loaded (%s); install opencv-python-headless", e)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
