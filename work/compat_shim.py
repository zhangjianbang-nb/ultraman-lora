"""Compat shim: diffusers 0.29 expects constants that transformers 5.x removed.

Import this BEFORE importing diffusers. Values match transformers 4.x.

    import compat_shim  # noqa: F401
    from diffusers import StableDiffusionXLPipeline
"""
import transformers.utils as tu

_DEFAULTS = {
    "FLAX_WEIGHTS_NAME": "flax_model.msgpack",
    "TF2_WEIGHTS_NAME": "tf_model.h5",
    "FLAX_WEIGHTS_INDEX_NAME": "flax_model.msgpack.index.json",
    "TF2_WEIGHTS_INDEX_NAME": "tf_model.h5.index.json",
    "FLAX_WEIGHTS_NAME_INDEX": "flax_model.msgpack.index.json",
}
for _k, _v in _DEFAULTS.items():
    if not hasattr(tu, _k):
        setattr(tu, _k, _v)
