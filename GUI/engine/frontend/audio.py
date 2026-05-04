"""Simple audio utilities for UI feedback sounds.

Uses pygame.mixer for cross-platform audio playback.
Falls back silently if audio is unavailable.

Audio is lazily initialized on first use to avoid:
- Initializing in processes that don't need audio (e.g., backend)
- Multiple initializations when using multiprocessing with 'spawn'
"""

import numpy as np

# Lazy initialization state
_audio_state = {
    "initialized": False,
    "available": False,
    "mixer": None,  # Will hold pygame.mixer module reference
    "cache": {},    # Sound cache (single instance)
}


def _ensure_audio_initialized():
    """Initialize audio system on first use. Returns True if audio is available."""
    if _audio_state["initialized"]:
        return _audio_state["available"]
    
    _audio_state["initialized"] = True
    
    try:
        import warnings
        warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
        warnings.filterwarnings("ignore", message=".*Your system is avx2 capable*")
        import os
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "1"
        import pygame.mixer
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
        _audio_state["mixer"] = pygame.mixer
        _audio_state["available"] = True
    except Exception:
        _audio_state["available"] = False
    
    return _audio_state["available"]


def _generate_blip(freq=880, duration_ms=30, volume=0.15, sample_rate=44100):
    """Generate a short blip/tick sound as a pygame Sound object."""
    n_samples = int(sample_rate * duration_ms / 1000)
    t = np.linspace(0, duration_ms / 1000, n_samples, dtype=np.float32)
    
    # Simple sine wave with quick fade envelope
    envelope = np.exp(-t * 40)  # fast decay
    wave = np.sin(2 * np.pi * freq * t) * envelope * volume
    
    # Convert to 16-bit PCM mono
    audio = (wave * 32767).astype(np.int16)
    
    # Create pygame Sound from buffer
    mixer = _audio_state["mixer"]
    sound = mixer.Sound(buffer=audio.tobytes())
    return sound


def _get_cached_sound(name, freq, duration_ms, volume):
    """Get or create cached Sound object."""
    if not _ensure_audio_initialized():
        return None
    
    cache = _audio_state["cache"]
    if name not in cache:
        try:
            cache[name] = _generate_blip(freq=freq, duration_ms=duration_ms, volume=volume)
        except Exception:
            cache[name] = None
    return cache[name]


def play_hover_in():
    """Play a short high-pitched blip for hover-in."""
    sound = _get_cached_sound("hover_in", freq=600, duration_ms=30, volume=0.08)
    if sound:
        sound.play()


def play_hover_out():
    """Play a short lower-pitched blip for hover-out."""
    sound = _get_cached_sound("hover_out", freq=400, duration_ms=20, volume=0.06)
    if sound:
        sound.play()


def play_click():
    """Play a click sound."""
    sound = _get_cached_sound("click", freq=600, duration_ms=40, volume=0.12)
    if sound:
        sound.play()
