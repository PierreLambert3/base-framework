POSTPROCESSING_BLOOM = 1
BLOOM_TINT           = (1.0, 0.01, 0.0)
# BLOOM_TINT           = (0.95, 0.01, 0.5)
POSTPROCESSING_NOISE = 0

ORANGE_YELLOW    = "#FFA500"
ORANGE_WHITE     = "#FADE9B"
ORANGE_RED       = "#FF4500"
ORANGE_DARK      = "#AD3D00"

PINK_NEON        = "#FF3557"
PINK_ELECTRIC    = "#EA00FF"
LAVENDER         = "#B855FF"
PURPLE_LIGHT     = "#9370DB"
PURPLE_DARK      = "#4B0082"

import numpy as np
def interpolate_color(color1, color2, factor: float) -> str:
    c1 = np.array([int(color1[i:i+2], 16) for i in (1, 3, 5)])
    c2 = np.array([int(color2[i:i+2], 16) for i in (1, 3, 5)])
    c_interp = (1 - factor) * c1 + factor * c2
    return '#' + ''.join(f'{int(c):02X}' for c in c_interp)

def brighten(color: str, factor: float) -> str:
    c = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
    c_bright = np.clip(c + factor * 255, 0, 255)
    return '#' + ''.join(f'{int(ci):02X}' for ci in c_bright)

def darken(color: str, factor: float) -> str:
    c = np.array([int(color[i:i+2], 16) for i in (1, 3, 5)])
    c_dark = np.clip(c - factor * 255, 0, 255)
    return '#' + ''.join(f'{int(ci):02X}' for ci in c_dark)

def transparent(color: str, alpha: float) -> str:
    a = int(alpha * 255)
    return color + f'{a:02X}'