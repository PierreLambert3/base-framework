# Wiki: wiki/03-backend.md (project-side backend, worker spawning)
# Related: wiki/01-architecture.md, wiki/02-communications.md,
#          wiki/06-worker-instances.md, wiki/08-extending-the-framework.md.
# Subclass `Custom_Backend` for project-specific behaviour; do not modify
# `GUI/engine/backend/logic.py` (the engine base class).

import time

from GUI.engine.backend.logic import Back_End
from GUI.engine.comms import Communications

from cuda_wrapper import CUDAManager

from worker.worker_instance import WorkerInstance, comms_prefix
from custom_worker  import CustomWorker


class Custom_Backend(Back_End):
    """
    Project-side backend. Lives in the main process and is responsible for:
      - exchanging messages with the frontend (which lives in its own process),
      - spawning `WorkerInstance` processes on demand,
      - relaying messages between the frontend and individual instances.

    Subclass this for project-specific behaviour. By default it understands
    one generic event from the frontend, `"launch worker instance"`, whose
    payload is `{"config": <opaque dict>, "instance_name_hint": <str|None>}`.
    """

    def __init__(self, multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict):
        super().__init__(multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict)

        # CUDA: one manager in the main (backend) process. Detects the device and
        # provides per-process configuration; the actual CUDA context for each
        # worker is created here (un-entered) and entered inside the child process.
        self.cuda_manager = CUDAManager(device_id=0, kernel_dir="kernels")

        # spawned worker processes
        self.worker_instance_names = []
        self.worker_processes      = []
        self.comms_instances       = []
        self.cuda_contexts         = []  # un-entered CUDAContext per instance
        self.data_stream_queues    = []  # tuples (front_to_back, back_to_front) from frontend's POV
        self.n_instances_created   = 0

        # default state
        self.simulation_chunk_size = 1

        # listeners
        self.build_listeners()

    # --------------------------------------------------------------- main loop
    def routine(self):
        # ask the frontend for the initial chunk size
        self.send("Q1: how many timesteps per simulation chunk", None)
        while self.running:
            time.sleep(0.25)
            self.process_messages()

    def process_messages(self):
        super().process_messages()
        # also drain each instance's comms (ACKs + instance-originated events)
        for comms in list(self.comms_instances):
            comms.process_messages()

    def build_listeners(self):
        self.add_listener("exit program",                                  self.exit_program)
        self.add_listener("RE1.1: how many timesteps per simulation chunk", self._set_simulation_chunk_size)
        self.add_listener("launch worker instance",                        self._handle_launch_worker_instance)
        self.add_listener("frontend ready for worker instance",           self._handle_frontend_ready_for_worker_instance)
        self.add_listener("worker instance deselected",                    self._handle_instance_deselected)
        self.add_listener("info for frontend",                             self._handle_info_from_instance)

    # --------------------------------------------------------------- shutdown
    def exit_program(self, data):
        super().exit_program(data)
        for name in self.worker_instance_names:
            self.send_to_instance(name, "exit program", None, require_ack=False)
        for proc in self.worker_processes:
            proc.join()

    # ------------------------------------------------------- chunk-size relay
    def _set_simulation_chunk_size(self, data):
        self.simulation_chunk_size = data
        for name in self.worker_instance_names:
            self.send_to_instance(name, "set simulation chunk size", data, require_ack=False)

    # ----------------------------------------------------- launching workers
    def _handle_launch_worker_instance(self, data):
        """`data` is `{"config": <dict>, "instance_name_hint": <str|None>}`."""
        data = data or {}
        config             = data.get("config", {})
        instance_name_hint = data.get("instance_name_hint")
        self.launch_worker_instance(config, instance_name_hint=instance_name_hint)

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

        # 2. data stream queues (instance --> frontend visualisation)
        data_stream_front_to_back = self.manager.Queue()
        data_stream_back_to_front = self.manager.Queue()

        # 3. CUDA context: created here (in the main process) but NOT entered.
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

        # 5. push the current chunk size to the new instance
        self.send_to_instance(instance_name, "set simulation chunk size", self.simulation_chunk_size, require_ack=False)

        # 6. inform the frontend
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
        return CustomWorker(
            instance_name, config,
            q_to_backend, q_from_backend,
            shared_dict,
            ds_q_f2b, ds_q_b2f,
            cuda_ctx,
        )

    # --------------------------------------------------------- routing helpers
    def send_to_instance(self, instance_name, event_name, event_data=None, require_ack=True):
        for i, name in enumerate(self.worker_instance_names):
            if name == instance_name:
                self.comms_instances[i].send(comms_prefix(instance_name) + event_name, event_data, needs_ack=require_ack)
                return
        print(f"FAILURE: could not send to instance '{instance_name}' because it does not exist!")

    def _handle_instance_deselected(self, data):
        """`data` is the instance_name."""
        self.send_to_instance(data, "instance deselected", None, require_ack=False)

    def _handle_frontend_ready_for_worker_instance(self, data):
        """`data` is the instance_name. Sent by frontend once it has allocated
        the visualisation resources for this instance."""
        self.send_to_instance(data, "frontend ready", None, require_ack=False)

    def _handle_info_from_instance(self, data):
        """A worker instance announces project-specific metadata once at
        startup. Forward to the frontend so the active page can use it."""
        self.send("worker instance info", data)
