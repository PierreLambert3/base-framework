import numpy as np

from worker.worker_instance import WorkerInstance
from worker.global_constants import TIMESTEP_DURATION_MS


class CustomWorker(WorkerInstance):
    """
    Example WorkerInstance subclass: simulates `n_points` 2D points moving
    with constant velocities and bouncing on the [0, 1]^2 walls.

    Required `config` keys:
      - "n_points": int   (e.g. 1_000 for "small", 1_000_000 for "big")

    Each simulation chunk advances the points and pushes their (x, y)
    positions to the frontend on the data stream channel using
    `needs_ack=False` (fire-and-forget).
    """

    def initialise(self):
        n_points = int(self.config.get("n_points", 1_000))
        rng = np.random.default_rng()
        self._n_points  = n_points
        self._positions = rng.uniform(0.0, 1.0, size=(n_points, 2)).astype(np.float32)
        # velocities: small, in normalized units per second
        self._velocities = (rng.uniform(-1.0, 1.0, size=(n_points, 2)) * 0.15).astype(np.float32)

    def _make_info_for_frontend(self):
        info = super()._make_info_for_frontend()
        info["n_points"] = self._n_points
        return info

    def run_simulation_chunk(self, chunk_size, selected_by_frontend, high_speed_mode):
        # 1. simulation
        for t in range(chunk_size):
            pass 

        # 2. update to frontend

        # Advance points by `chunk_size` timesteps
        dt = chunk_size * (TIMESTEP_DURATION_MS / 1000.0)
        self._positions += self._velocities * dt

        # Bounce off [0, 1] walls (mirror position + flip velocity component)
        for axis in (0, 1):
            below = self._positions[:, axis] < 0.0
            above = self._positions[:, axis] > 1.0
            if below.any():
                self._positions[below, axis] = -self._positions[below, axis]
                self._velocities[below, axis] = -self._velocities[below, axis]
            if above.any():
                self._positions[above, axis] = 2.0 - self._positions[above, axis]
                self._velocities[above, axis] = -self._velocities[above, axis]

        # Stream positions to the frontend (fire-and-forget)
        self.data_stream_comms.send(
            "data stream: positions",
            {"positions": self._positions},
            needs_ack=False,
        )
