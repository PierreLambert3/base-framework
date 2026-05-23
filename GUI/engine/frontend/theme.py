POSTPROCESSING_BLOOM = 1
BLOOM_TINT           = (1.0, 0.0, -0.1) # amber
# BLOOM_TINT           = (0.8, 0.1, 0.02) # amber (less intense)
# BLOOM_TINT           = (-0.1, -0.99,  0.83) # ultraviolet
# BLOOM_TINT           = (-0.1, -0.6, -0.2) # wierd
# BLOOM_TINT           = (-0.1, -0.8, 0.1) # pink neon

POINTS_MODE          = 1


POSTPROCESSING_NOISE = 0

AMBER            = "#FF7300"
AMBER2           = "#FA5908"
ORANGE_YELLOW    = "#FFA500"
ORANGE_WHITE     = "#FADE9B"
ORANGE_RED       = "#FF4500"
ORANGE_DARK      = "#470500C1"
BONE             = "#FDECCE"

PINK_NEON        = "#FF3557"
PINK_ELECTRIC    = "#EA00FF"
LAVENDER         = "#B855FF"
PURPLE_LIGHT     = "#9370DB"    
PURPLE_DARK      = "#4B0082"
BLUE_WIERDNESS   = "#0044FF"

TRANSLUCENT_DARK_HIGHLIGHT = "#673D2D1D"
DARK_HIGHLIGHT = "#90370047"

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

def to_rgba(color: str, new_alpha: float = None) -> tuple:
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    if new_alpha is not None:
        a = new_alpha
    elif len(color) == 9:
        a = int(color[7:9], 16) / 255.0
    else:
        a = 1.0
    return (r, g, b, a)

def to_rgb(color: str) -> tuple:
    r = int(color[1:3], 16) / 255.0
    g = int(color[3:5], 16) / 255.0
    b = int(color[5:7], 16) / 255.0
    return (r, g, b)

def brighten_tuple(colour, factor):
    r, g, b = colour[0], colour[1], colour[2]
    a = colour[3] if len(colour) > 3 else 1.0
    return (min(r * factor, 1.0), min(g * factor, 1.0), min(b * factor, 1.0), a)

def darken_tuple(colour, factor):
    r, g, b = colour[0], colour[1], colour[2]
    a = colour[3] if len(colour) > 3 else 1.0
    return (max(r - factor, 0.0), max(g - factor, 0.0), max(b - factor, 0.0), a)

def K_random_colours(K):
    return ["#" + ''.join(np.random.choice(list('0123456789ABCDEF'), size=6)) for _ in range(K)]

