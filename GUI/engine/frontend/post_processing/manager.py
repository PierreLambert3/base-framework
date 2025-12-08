from typing import Any
from bloom.bloom_pass import BloomPass

def attach_bloom(renderer: Any, **kwargs) -> BloomPass:
    """
    Attach a BloomPass to the renderer's effect chain.

    kwargs can include: depth, threshold, knee, strength, color_select,
    color_center, color_sigma, preferred_format.
    """
    bloom = BloomPass(**kwargs)
    renderer.effect_passes = (*renderer.effect_passes, bloom)
    return bloom
