# Wiki: wiki/06-worker-instances.md (canonical GPU worker template),
#       wiki/07-cuda-wrapper.md (kernel + memory APIs used here),
#       wiki/09-example-walkthrough.md (end-to-end trace of this worker).
# This is the reference example: copy & edit it for new project workers.

import time
import numpy as np

from cuda_wrapper import LaunchConfig1D, DeviceProperties

from GUI.engine.worker.worker_instance import WorkerInstance
from GUI.engine.worker.global_constants import TIMESTEP_DURATION_MS


class _Kernels:
    """
    All CUDA kernels used by `CustomWorker`, with their pre-computed launch
    configurations. Exposes one Python method per logical operation so that
    `CustomWorker` does not have to know about kernel-level details.
    """

    def __init__(self, worker, cuda_ctx):
        self.props = DeviceProperties()

        # One thread per simulated point.
        self._update_positions = cuda_ctx.get_kernel(
            "update_positions", "update_positions",
            LaunchConfig1D(self.props, n_workers=worker._n_points),
        )

    def update_positions(self, stream, worker, n_steps):
        """Advance all points by `n_steps` timesteps on the GPU."""
        self._update_positions.launch(
            stream,
            worker._positions_gpu,
            worker._velocities_gpu,
            worker._n_points_u32,
            worker._dt_per_step,
            np.uint32(n_steps),
        )


class _Streams:
    """
    CUDA streams used by `CustomWorker`. Currently a single compute stream;
    additional streams (e.g. for async D->H transfers) can be added here.
    """

    def __init__(self, cuda_ctx):
        self.compute = cuda_ctx.stream()

    def sync_all(self):
        self.compute.sync()


class CustomWorker(WorkerInstance):
    """
    Example WorkerInstance subclass: simulates `n_points` 2D points moving
    with constant velocities and bouncing on the [0, 1]^2 walls.

    Positions and velocities live on the GPU; a CUDA kernel
    (`kernels/update_positions.cu`) advances them by `chunk_size` timesteps
    per simulation chunk. Positions are copied back to the host only just
    before being streamed to the frontend.

    Required `config` keys:
      - "n_points": int   (e.g. 1_000 for "small", 1_000_000 for "big")
    """

    def initialise(self):
        # `self.cuda_ctx` has already been entered by `WorkerInstance.routine`.
        n_points = int(self.config.get("n_points", 1_000))
        self.EMA_iterations_per_second = 1.0
        self.prev_chunk_start_time = time.time()
        self.prev_chunk_size = 10

        positions  = np.random.uniform(0.0, 1.0, size=(n_points, 2)).astype(np.float32)
        velocities = (np.random.uniform(-1.0, 1.0, size=(n_points, 2)) * 0.15).astype(np.float32)

        # GPU-resident state.
        self._n_points       = n_points
        self._positions_gpu  = self.cuda_ctx.m(positions)
        self._velocities_gpu = self.cuda_ctx.m(velocities)

        # Pre-built scalar kernel arguments (numpy-typed for type safety).
        self._n_points_u32 = np.uint32(n_points)
        self._dt_per_step  = np.float32(TIMESTEP_DURATION_MS / 1000.0)

        # CUDA streams + kernels.
        self.streams = _Streams(self.cuda_ctx)
        self.kernels = _Kernels(self, self.cuda_ctx)

        # Transport for positions: either a per-chunk queue payload (default)
        # or a shared-memory block (`use_shared_memory: True` in config). The
        # shared block is owned by this worker; its name is shipped to the
        # frontend via `_make_info_for_frontend`.
        self._use_shared_memory = bool(self.config.get("use_shared_memory", False))
        self._frame_id          = 0
        if self._use_shared_memory:
            # `data_stream_comms` owns the SHM block (matches the channel the
            # frontend listens on for the doorbell notification).
            self._positions_host = self.data_stream_comms.create_shared_array(
                "positions", (n_points, 2), np.float32,
            )
        else:
            # Pre-allocated host buffer for D->H transfers (avoids per-chunk allocation).
            self._positions_host = np.empty((n_points, 2), dtype=np.float32)

    def _on_exit(self, data):
        # Make sure all GPU work has completed before the context is released.
        self.streams.sync_all()

    def _make_info_for_frontend(self):
        info = super()._make_info_for_frontend()
        info["n_points"] = self._n_points
        if self._use_shared_memory:
            info["shared_memory"] = {
                "positions": self.data_stream_comms.get_shared_array_info("positions"),
            }
        return info

    def run_simulation_chunk(self, chunk_size, selected_by_frontend, high_speed_mode):
        
        # 1. monitoring of iteration rate
        tic = time.time()
        chunk_time_taken = tic - self.prev_chunk_start_time
        steps_per_second = self.prev_chunk_size / chunk_time_taken if chunk_time_taken > 0 else 0.0
        self.EMA_iterations_per_second *= 0.8
        self.EMA_iterations_per_second += (1 - 0.8) * steps_per_second
        self.prev_chunk_start_time = tic
        self.prev_chunk_size       = chunk_size
        
        # 2. the simulation chunk
        for _ in range(chunk_size):
            time.sleep(0.0003)
            self.kernels.update_positions(self.streams.compute, self, chunk_size)

        # 3. Copy positions back to the host (synchronous w.r.t. the stream).
        self._positions_gpu.to_host(out=self._positions_host, stream=self.streams.compute)
        self.streams.compute.sync()

        # 4. Notify the frontend. With shared memory, the host buffer IS the
        #    inter-process buffer, so we only ring a doorbell with a frame id.
        #    With queues, we ship the array itself (fire-and-forget snapshot).
        if self._use_shared_memory:
            self._frame_id += 1
            self.data_stream_comms.send(
                "data stream: positions ready",
                {"frame_id": self._frame_id},
                needs_ack=True, # ensure the frontend has acked the previous message, if not, the (new) data will be sent as soon as the ack is received
            )
        else:
            self.data_stream_comms.send(
                "data stream: positions",
                {"positions": self._positions_host},
                needs_ack=True, # ensure the frontend has acked the previous message, if not, the (new) data will be sent as soon as the ack is received
            )

        # 5. if selected, send the iteration rate to the frontend for display in the overlay
        if self.selected_by_frontend:
            self.data_stream_comms.send(
                "data stream: iteration rate",
                {"iterations_per_second": round(self.EMA_iterations_per_second, 1)},
                needs_ack=True,
            )
