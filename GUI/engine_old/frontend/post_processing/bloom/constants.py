# Bloom effect global configuration and defaults

BLOOM_DEPTH = 5  # Pyramid levels for downsample/upsample
BLOOM_HDR_FORMAT = "rgba16float"  # Use HDR format if available
BLOOM_DEFAULTS = dict(
    threshold=1.0,
    knee=0.5,
    strength=0.8,
    color_select=False,
    color_center=(1.0, 1.0, 1.0),
    color_sigma=0.35,
)

# Kawase offsets for blur (8-tap)
KAWASE_OFFSETS = [
    (1, 1), (-1, 1), (1, -1), (-1, -1),
    (2, 0), (-2, 0), (0, 2), (0, -2),
]
