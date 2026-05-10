import multiprocessing
import time
import numpy as np

from worker.global_constants import TIMESTEP_DURATION_MS

from GUI.engine.comms import _Listeners, Communications


def comms_prefix(instance_name):
    return "<" + instance_name + "> "


class WorkerInstance:
    """
    Generic worker process scaffolding.

    A WorkerInstance runs in its own OS process (spawned from the backend) and
    exchanges messages with the backend over a dedicated pair of queues, plus a
    second pair of queues used for high-throughput data streaming towards the
    frontend for visualisation purposes (similar to TCP vs UDP in philosophy).

    Project-specific behaviour is added by subclassing and overriding the
    hooks: `initialise`, `run_chunk`, `_on_chunk_size_changed`, `_on_exit`,
    and `_make_info_for_frontend`.
    """

    def __init__(self,
                 instance_name,
                 config,
                 queue_to_backend,
                 queue_from_backend,
                 shared_dict,
                 data_stream_queue_front_to_back,
                 data_stream_queue_back_to_front):
        self.name   = instance_name
        self.config = config  # opaque, project-defined dict

        self.frontend_ready_for_simulation = False

        # state
        self.simulation_chunk_size = 1
        self.high_speed_mode       = False
        self.running               = True

        # visualisation state
        self.selected_by_frontend = False

        # main communications (backend <-> instance)
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_backend, queue_to_backend, shared_dict, self.listeners)

        # data stream communications (instance --> frontend, separate channel)
        self.data_stream_listeners = _Listeners()
        self.data_stream_comms     = Communications(data_stream_queue_front_to_back, data_stream_queue_back_to_front, shared_dict, self.data_stream_listeners)

        # listeners
        self.build_listeners()

    # ------------------------------------------------------------------ hooks
    def initialise(self):
        """Override to set up project-specific resources (e.g. CUDA context,
        GPU buffers, models). Called once at the start of `routine()` inside
        the spawned process."""
        pass

    def run_simulation_chunk(self, chunk_size, selected_by_frontend, high_speed_mode):
        """Override to advance the simulation by `chunk_size` steps.
        Default implementation is a placeholder that just sleeps so the
        framework can be exercised end-to-end without any project logic."""
        time.sleep(np.random.uniform(0.01, 0.02) * chunk_size)

    def _on_chunk_size_changed(self, chunk_size, high_speed_mode):
        """Override to react to the frontend changing the simulation chunk size."""
        pass

    def _on_exit(self, data):
        """Override to release project-specific resources before queues are torn down."""
        pass

    def _make_info_for_frontend(self):
        """Override (and call super().) to add project-specific metadata sent
        to the frontend once at startup."""
        return {"instance name": self.name}

    # -------------------------------------------------------------- main loop
    def routine(self):
        # 1. project-specific setup
        self.initialise()

        # 2. announce ourselves and wait until the frontend is ready
        self.comms.send("info for frontend", self._make_info_for_frontend())
        while not self.frontend_ready_for_simulation:
            time.sleep(0.1)
            self.process_messages()

        # 3. main simulation loop
        EMA_simulated_seconds_per_real_second = 1.0
        while self.process_messages():
            tic = time.time()
            self.run_simulation_chunk(self.simulation_chunk_size, self.selected_by_frontend, self.high_speed_mode)

            if self.selected_by_frontend:
                elapsed_s = time.time() - tic
                if elapsed_s > 0:
                    EMA_simulated_seconds_per_real_second = (
                        0.95 * EMA_simulated_seconds_per_real_second
                        + 0.05 * ((self.simulation_chunk_size * TIMESTEP_DURATION_MS / 1000.0) / elapsed_s)
                    )
                self.comms.send("seconds per real second update", {
                    "sec per sec": EMA_simulated_seconds_per_real_second,
                    "chunk size":  self.simulation_chunk_size,
                })
            # tiny sleep to reduce contention on the data stream queues
            time.sleep(0.06)

    # ----------------------------------------------------------- wiring/utils
    def build_listeners(self):
        prefix = comms_prefix(self.name)
        self.add_listener(prefix + "set simulation chunk size", self._set_simulation_chunk_size)
        self.add_listener(prefix + "frontend ready",            self._handle_frontend_ready)
        self.add_listener(prefix + "exit program",              self.exit_program)

    def exit_program(self, data):
        self._on_exit(data)
        self.data_stream_comms.empty_queues()
        self.comms.empty_queues()
        self.comms.cancel_join_threads()
        self.data_stream_comms.cancel_join_threads()
        self.running = False

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def process_messages(self):
        self.comms.process_messages()
        self.data_stream_comms.process_messages()
        return self.running

    def start(self):
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = f"WorkerInstance Process: {self.name}",
            daemon = False,
        )
        process.start()
        return process

    # -------------------------------------------------------- default handlers
    def _set_simulation_chunk_size(self, data):
        self.simulation_chunk_size = data
        self.high_speed_mode       = self.simulation_chunk_size > 200
        self._on_chunk_size_changed(self.simulation_chunk_size, self.high_speed_mode)

    def _handle_frontend_ready(self, data):
        self.frontend_ready_for_simulation = True
