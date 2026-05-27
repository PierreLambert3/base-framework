# Wiki: wiki/03-backend.md (project-side backend, worker spawning)
# Related: wiki/01-architecture.md, wiki/02-communications.md,
#          wiki/06-worker-instances.md, wiki/08-extending-the-framework.md.
# Subclass `Custom_Backend` for project-specific behaviour; do not modify
# `GUI/engine/backend/logic.py` (the engine base class).

import time

from GUI.engine.backend.logic import Back_End
from cuda_wrapper import CUDAManager
from custom_worker import CustomWorker


class Custom_Backend(Back_End):
    """
    Project-side backend. Responsible for CUDA initialisation, providing the
    concrete worker type, and hooking into the worker-lifecycle events defined
    by the engine base class (Back_End).

    Override the hook methods below to add project-specific behaviour at each
    worker-lifecycle event. By default they just delegate to the base
    implementation.
    """

    def __init__(self, multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict,
                 queue_from_main=None, queue_to_main=None, max_worker_instances=8):
        super().__init__(multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict,
                         queue_from_main=queue_from_main, queue_to_main=queue_to_main,
                         max_worker_instances=max_worker_instances)
        self.build_listeners()

    # --------------------------------------------------------------- main loop

    def routine(self):
        if self.cuda_manager is None:
            self.cuda_manager = CUDAManager(device_id=0, kernel_dir="kernels")

        # ask the frontend for the initial chunk size
        self.send("Q1: how many timesteps per simulation chunk", None)
        while self.running:
            time.sleep(0.25)
            self.process_messages()

    def build_listeners(self):
        super().build_listeners()
        self.add_listener("RE1.1: how many timesteps per simulation chunk", self._set_simulation_chunk_size)

    # --------------------------------------------------------- worker factory

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

    # ------------------------------------------------ worker-lifecycle hooks
    # Each method below is a hook: add project-specific logic before/after
    # the super() call as needed.

    def exit_program(self, data):
        super().exit_program(data)

    def _handle_launch_worker_instance(self, data):
        super()._handle_launch_worker_instance(data)

    def _handle_deallocate_worker_instance(self, data):
        super()._handle_deallocate_worker_instance(data)

    def _handle_frontend_ready_for_worker_instance(self, data):
        super()._handle_frontend_ready_for_worker_instance(data)

    def _handle_instance_deselected(self, data):
        super()._handle_instance_deselected(data)

    def _handle_instance_selected(self, data):
        super()._handle_instance_selected(data)

    def _handle_info_from_instance(self, data):
        super()._handle_info_from_instance(data)
