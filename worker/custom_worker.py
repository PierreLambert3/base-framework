# Wiki: wiki/06-worker-instances.md (canonical GPU worker template),
#       wiki/07-cuda-wrapper.md (kernel + memory APIs used here),
#       wiki/09-example-walkthrough.md (end-to-end trace of this worker).
# This is the reference example: copy & edit it for new project workers.

import numpy as np

from cuda_wrapper import LaunchConfig1D, DeviceProperties

from worker.worker_instance import WorkerInstance
from worker.global_constants import TIMESTEP_DURATION_MS


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
        rng = np.random.default_rng()

        positions  = rng.uniform(0.0, 1.0, size=(n_points, 2)).astype(np.float32)
        velocities = (rng.uniform(-1.0, 1.0, size=(n_points, 2)) * 0.15).astype(np.float32)

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

        # Pre-allocated host buffer for D->H transfers (avoids per-chunk allocation).
        self._positions_host = np.empty((n_points, 2), dtype=np.float32)

    def _on_exit(self, data):
        # Make sure all GPU work has completed before the context is released.
        self.streams.sync_all()

    def _make_info_for_frontend(self):
        info = super()._make_info_for_frontend()
        info["n_points"] = self._n_points
        return info

    def run_simulation_chunk(self, chunk_size, selected_by_frontend, high_speed_mode):
        # 1. Advance positions on the GPU (one launch per chunk).
        self.kernels.update_positions(self.streams.compute, self, chunk_size)

        # 2. Copy positions back to the host (synchronous w.r.t. the stream).
        self._positions_gpu.to_host(out=self._positions_host, stream=self.streams.compute)
        self.streams.compute.sync()

        # 3. Stream positions to the frontend (fire-and-forget). The queue
        #    serialises the array, so a fresh snapshot is sent each chunk.
        self.data_stream_comms.send(
            "data stream: positions",
            {"positions": self._positions_host},
            needs_ack=False,
        )
