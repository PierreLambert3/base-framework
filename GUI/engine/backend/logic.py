# Wiki: wiki/03-backend.md (engine base class for the backend process).
# Engine file -- do not edit for project work; subclass `Back_End`
# (see `GUI/backend.py::Custom_Backend`).

import multiprocessing
import time
from GUI.engine.comms import _Listeners, Communications
from GUI.engine.worker.worker_instance import comms_prefix

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

        # CUDA manager (None until subclass assigns it, e.g. in routine()).
        # launch_worker_instance() calls self.cuda_manager.create_context(...)
        # so the subclass must initialise this before any workers are spawned.
        self.cuda_manager = None

        # Worker process bookkeeping.
        self.worker_instance_names     = []
        self.worker_processes          = []
        self.comms_instances           = []
        self.cuda_contexts             = []   # un-entered CUDAContext per instance
        self.data_stream_queues        = []   # tuples (front_to_back, back_to_front) from frontend's POV
        self.data_stream_queue_indices = []   # corresponding pool slot indices
        self.n_instances_created       = 0

        # Simulation chunk size broadcast to all workers when it changes.
        self.simulation_chunk_size = 1

        # Name of the currently auto-selected worker instance (set on first spawn).
        self._selected_instance_name = None

    def routine(self):
        pass

    def build_listeners(self):
        """Register all base worker-lifecycle listeners.
        Derived classes must call super().build_listeners() to include these."""
        self.add_listener("exit program",                       self.exit_program)
        self.add_listener("launch worker instance",             self._handle_launch_worker_instance)
        self.add_listener("deallocate worker instance",         self._handle_deallocate_worker_instance)
        self.add_listener("frontend ready for worker instance", self._handle_frontend_ready_for_worker_instance)
        self.add_listener("worker instance deselected",         self._handle_instance_deselected)
        self.add_listener("info for frontend",                  self._handle_info_from_instance)

    def exit_program(self, data):
        for name in list(self.worker_instance_names):
            self.send_to_instance(name, "exit program", None, require_ack=False)
        for proc in self.worker_processes:
            proc.join()
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
        for comms in list(self.comms_instances):
            comms.process_messages()

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

    # ----------------------------------------------------- simulation chunk size
    def _set_simulation_chunk_size(self, data):
        self.simulation_chunk_size = data
        for name in self.worker_instance_names:
            self.send_to_instance(name, "set simulation chunk size", data, require_ack=False)

    # ------------------------------------------------------- worker management
    def launch_worker_instance(self, config, instance_name_hint=None):
        """Spawn a new WorkerInstance process. Can be called directly from a
        subclass without going through the frontend message."""
        self.n_instances_created += 1
        instance_name = (
            f"{instance_name_hint} {self.n_instances_created}"
            if instance_name_hint
            else f"WorkerInstance {self.n_instances_created}"
        )

        # 1. queues backend <-> instance
        backend_to_subprocess = self.multiprocessing_context.Queue()
        subprocess_to_backend = self.multiprocessing_context.Queue()
        comms_to_instance     = Communications(subprocess_to_backend, backend_to_subprocess, self.comms.shared, self.listeners)

        # 2. data stream queues (instance --> frontend visualisation) — drawn from
        #    the pre-allocated pool so no Manager is needed in the subprocess.
        acquired = self._acquire_queue_pair()
        if acquired is None:
            print(f"WARNING: queue pool exhausted — cannot launch '{instance_name}'")
            self.send("worker instance launch refused", {"reason": "queue pool exhausted"})
            return None
        queue_index, data_stream_front_to_back, data_stream_back_to_front = acquired

        # 3. CUDA context: created here (in the backend process) but NOT entered.
        #    The child process will enter it inside its own routine().
        cuda_ctx = self.cuda_manager.create_context(uses_pytorch=False)

        # 4. instantiate and start the process
        instance = self._make_worker_instance(
            instance_name, config,
            subprocess_to_backend, backend_to_subprocess,
            self.comms.shared,
            data_stream_front_to_back, data_stream_back_to_front,
            cuda_ctx,
        )
        process = instance.start()

        # 5. bookkeeping
        self.worker_processes.append(process)
        self.worker_instance_names.append(instance_name)
        self.comms_instances.append(comms_to_instance)
        self.cuda_contexts.append(cuda_ctx)
        self.data_stream_queues.append((data_stream_front_to_back, data_stream_back_to_front))
        self.data_stream_queue_indices.append(queue_index)

        # 6. push the current chunk size to the new instance
        self.send_to_instance(instance_name, "set simulation chunk size", self.simulation_chunk_size, require_ack=False)

        # 7. inform the frontend
        self.send("new worker instance created", {
            "instance name":      instance_name,
            "config":             config,
            "data_stream_queues": (data_stream_back_to_front, data_stream_front_to_back),
        })
        return instance_name

    def _make_worker_instance(self,
                              instance_name, config,
                              q_to_backend, q_from_backend,
                              shared_dict,
                              ds_q_f2b, ds_q_b2f,
                              cuda_ctx):
        """Override in subclass to return a concrete WorkerInstance."""
        raise NotImplementedError("Back_End._make_worker_instance() must be implemented in the derived class.")

    def deallocate_worker_instance(self, instance_name):
        try:
            i = self.worker_instance_names.index(instance_name)
        except ValueError:
            print(f"WARNING: deallocate_worker_instance('{instance_name}') — instance not found")
            return
        self.send_to_instance(instance_name, "exit program", None, require_ack=False)
        self.worker_processes[i].join(timeout=5.0)
        if self.worker_processes[i].is_alive():
            self.worker_processes[i].terminate()
            self.worker_processes[i].join()
        self._release_queue_pair(self.data_stream_queue_indices[i])
        if self._selected_instance_name == instance_name:
            self._selected_instance_name = None
        self.worker_instance_names.pop(i)
        self.worker_processes.pop(i)
        self.comms_instances.pop(i)
        self.cuda_contexts.pop(i)
        self.data_stream_queues.pop(i)
        self.data_stream_queue_indices.pop(i)

    def send_to_instance(self, instance_name, event_name, event_data=None, require_ack=True):
        for i, name in enumerate(self.worker_instance_names):
            if name == instance_name:
                self.comms_instances[i].send(comms_prefix(instance_name) + event_name, event_data, needs_ack=require_ack)
                return
        print(f"FAILURE: could not send to instance '{instance_name}' because it does not exist!")

    # ------------------------------------------ base worker-lifecycle handlers
    def _handle_launch_worker_instance(self, data):
        """`data` is `{"config": <dict>, "instance_name_hint": <str|None>}`."""
        data = data or {}
        config             = data.get("config", {})
        instance_name_hint = data.get("instance_name_hint")
        self.launch_worker_instance(config, instance_name_hint=instance_name_hint)

    def _handle_deallocate_worker_instance(self, data):
        self.deallocate_worker_instance(data)

    def _handle_frontend_ready_for_worker_instance(self, data):
        """`data` is the instance_name. Sent by the frontend once it has
        allocated the visualisation resources for this instance."""
        instance_name = data
        self.send_to_instance(instance_name, "frontend ready", None, require_ack=False)
        # Auto-select the first worker instance for which the frontend is ready.
        if self._selected_instance_name is None:
            self._selected_instance_name = instance_name
            self.send_to_instance(instance_name, "instance selected", None, require_ack=False)

    def _handle_instance_deselected(self, data):
        """`data` is the instance_name."""
        if self._selected_instance_name == data:
            self._selected_instance_name = None
        self.send_to_instance(data, "instance deselected", None, require_ack=False)

    def _handle_instance_selected(self, data):
        """`data` is the instance_name."""
        self._selected_instance_name = data
        self.send_to_instance(data, "instance selected", None, require_ack=False)

    def _handle_info_from_instance(self, data):
        """A worker instance announces project-specific metadata once at
        startup. Forward to the frontend so the active page can use it."""
        self.send("worker instance info", data)

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