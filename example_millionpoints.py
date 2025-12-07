import time
import threading
import numpy as np
import pygfx as gfx
from rendercanvas.auto import RenderCanvas, loop

"""
Million Dots Performance Test

Renders 1,000,000 points whose positions are re-randomized every frame.
New positions are generated in a background thread while the current
frame renders (double buffering approach).
Displays FPS in the console and (if supported) updates the window title.
"""

# Configuration
NUM_POINTS = 1_000_000
# NUM_POINTS = 1000
RANDOM_RANGE = (-1.0, 1.0)  # xyz uniform range
PRINT_FPS_INTERVAL = 1.0    # seconds
POINT_SIZE = 2.0            # in screen pixels

# Canvas / renderer / scene
canvas = RenderCanvas(title="Million Dots (initializing)")
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()

# Camera (simple perspective)
camera = gfx.PerspectiveCamera(60, int(16 / 9))
camera.local.position = (0, 0, 4)
scene.add(camera)

# Geometry buffers (double buffer for generation)
current_positions = np.empty((NUM_POINTS, 3), dtype=np.float32)
next_positions    = np.empty_like(current_positions)
rng = np.random.Generator(np.random.PCG64())

# Initialize first frame positions
current_positions[:] = rng.uniform(RANDOM_RANGE[0], RANDOM_RANGE[1], size=current_positions.shape).astype(np.float32)

# Use explicit Buffer + send_data pattern (more explicit GPU upload control)
import wgpu
positions_buf = gfx.Buffer(
    nitems=NUM_POINTS,
    nbytes=NUM_POINTS * 3 * 4,
    format="3xf4",
    usage=wgpu.BufferUsage.COPY_DST,
)
# Upload initial data
positions_buf.send_data(0, current_positions)

geometry = gfx.Geometry(positions=positions_buf)
material = gfx.PointsMaterial(size=int(POINT_SIZE), color=(1, 1, 1, 1))
points   = gfx.Points(geometry, material)
scene.add(points)

# Threading primitives for background generation
start_gen_event  = threading.Event()
next_ready_event = threading.Event()
shutdown_event   = threading.Event()


def _generator_loop():
    while not shutdown_event.is_set():
        # Wait until main thread signals start of new generation
        signaled = start_gen_event.wait(timeout=0.5)
        if not signaled:
            continue
        start_gen_event.clear()
        if shutdown_event.is_set():
            break
        # Fill next buffer with new random coordinates
        next_positions[:] = rng.uniform(RANDOM_RANGE[0], RANDOM_RANGE[1], size=next_positions.shape).astype(np.float32)
        next_ready_event.set()


worker = threading.Thread(target=_generator_loop, name="PositionGenerator", daemon=True)
worker.start()

# Kick off first generation for frame+1
start_gen_event.set()

# FPS tracking
_last_fps_report = time.time()
_frame_counter = 0
_fps = 0.0


def _update_fps():
    global _last_fps_report, _frame_counter, _fps
    _frame_counter += 1
    now = time.time()
    elapsed = now - _last_fps_report
    if elapsed >= PRINT_FPS_INTERVAL:
        _fps = _frame_counter / elapsed
        _frame_counter = 0
        _last_fps_report = now
        # Console output
        print(f"FPS: {_fps:.1f}")
        # Try updating window title
        try:
            canvas.set_title(f"Million Dots - {_fps:.1f} FPS")
        except Exception:
            pass


def draw():
    global current_positions, next_positions
    # If background thread finished a new buffer, copy into current and start another generation
    if next_ready_event.is_set():
        next_ready_event.clear()
        # Upload freshly generated data (next_positions) directly
        geometry.positions.send_data(0, next_positions)
        # Swap buffers so current_positions always reflects what is on GPU
        current_positions, next_positions = next_positions, current_positions
        # Start generating the next frame's data
        start_gen_event.set()

    renderer.render(scene, camera)
    _update_fps()
    # Schedule next frame for continuous animation
    canvas.request_draw(draw)


# Ensure graceful shutdown on exit (best effort)
import atexit


def _shutdown():
    shutdown_event.set()
    start_gen_event.set()
    try:
        worker.join(timeout=0.5)
    except Exception:
        pass


atexit.register(_shutdown)

canvas.request_draw(draw)
loop.run()
