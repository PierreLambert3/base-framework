# Wiki: wiki/03-backend.md (engine base class for the backend process).
# Engine file -- do not edit for project work; subclass `Back_End`
# (see `GUI/backend.py::Custom_Backend`).

import multiprocessing
import time
from GUI.engine.comms import _Listeners, Communications

DEAD_AFTER_S = 5.0

REGISTER_SHARED_ARRAY_FROM_MAIN = "register shared array (main->backend)"

class Back_End:
    def __init__(self, multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict,
                 queue_from_main=None, queue_to_main=None, max_worker_instances=8):
        self.multiprocessing_context = multiprocessing_context
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_frontend, queue_to_frontend, shared_dict, self.listeners)
        self.running   = True

        # Pre-allocated pool of data-stream queue pairs.  Created here (before
        # the process is spawned) using plain ctx.Queue() so they survive
        # pickling without needing a SyncManager in the subprocess.
        # Each slot is (q_front_to_back, q_back_to_front).
        self._ds_queue_pool      = [
            (manager.Queue(), manager.Queue()) for _ in range(max_worker_instances)
        ]
        self._ds_queue_pool_used = [False] * max_worker_instances

        # Optional main<->backend channel (async backend mode). When the
        # backend runs in its own process, the main process talks to it via
        # this second Communications, which reuses the same `_Listeners`
        # registry so `add_listener(...)` works uniformly for events
        # originating from either the frontend or the main process.
        # Event names from main must not collide with frontend-originated ones.
        self.main_comms         = None
        self.main_shared_arrays = {} # key -> ndarray view (attached from main)
        if queue_from_main is not None and queue_to_main is not None:
            self.main_comms = Communications(queue_from_main, queue_to_main, shared_dict, self.listeners)
            self.listeners.add(REGISTER_SHARED_ARRAY_FROM_MAIN, self._on_register_shared_array_from_main)

    def routine(self):
        pass

    def build_listeners(self):
        raise Exception("--- Back_End.build_listeners() should be implemented in the derived class! ---")

    def exit_program(self, data):
        self.running = False
        self.comms.send("exit program", None, needs_ack=False)
        self.comms.cancel_join_threads()
        if self.main_comms is not None:
            self.main_comms.send("exit program", None, needs_ack=False)
            self.main_comms.cancel_join_threads()

    def send(self, event_name, event_data=None, min_interval_s=0.0):
        self.comms.send(event_name, event_data)

    def send_to_main(self, event_name, event_data=None, needs_ack=True):
        if self.main_comms is None:
            print(f"FAILURE: send_to_main('{event_name}') ignored -- main_comms is not configured (blocking backend mode).")
            return
        self.main_comms.send(event_name, event_data, needs_ack=needs_ack)

    def process_messages(self):
        self.comms.process_messages()
        if self.main_comms is not None:
            self.main_comms.process_messages()

    def update_shared_dict(self, key, value):
        self.comms.shared.set(key, value)

    def process_shared_dict(self):
        raise Exception("--- Back_End.process_shared_dict() should be implemented in the derived class! ---")

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def read_shared_dict(self, key, default=None):
        return self.comms.shared.get(key, default=default)

    def get_main_shared_array(self, key):
        """Return the numpy view for a shared array registered by the main process."""
        return self.main_shared_arrays.get(key)

    def _on_register_shared_array_from_main(self, data):
        """Default handler: main side allocated a shared-memory array and
        sent us its descriptor. Attach to it and store the view."""
        key   = data["key"]
        name  = data["name"]
        shape = tuple(data["shape"])
        dtype = data["dtype"]
        arr = self.main_comms.attach_shared_array(key, name, shape, dtype)
        self.main_shared_arrays[key] = arr

    # ------------------------------------------------- data-stream queue pool
    def _acquire_queue_pair(self):
        for i, used in enumerate(self._ds_queue_pool_used):
            if not used:
                self._ds_queue_pool_used[i] = True
                q_f2b, q_b2f = self._ds_queue_pool[i]
                return i, q_f2b, q_b2f
        return None

    def _release_queue_pair(self, index):
        self._ds_queue_pool_used[index] = False

    def start(self):
        """Spawn the backend in its own process (async backend mode)."""
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = "Backend Process",
            daemon = False,
        )
        process.start()
        return process
    
    def _drain_data_stream_queues(self, queues=None, comms=None):
        if queues is not None:
            for queue in queues:
                while not queue.empty():
                    try:
                        _ = queue.get_nowait()
                    except:
                        break
        if comms is not None:
            comms.empty_queues()