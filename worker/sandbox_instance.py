import multiprocessing
import numpy as np

from worker.global_constants import TIMESTEP_DURATION_MS

from GUI.engine.comms import _Listeners, Communications

def comms_prefix(instance_name):
    return "<" + instance_name + "> "

class SandboxInstance():
    def __init__(self, cuda_context, instance_name, environment_type, brain_type, has_reservoir, learning_type, queue_to_backend, queue_from_backend, shared_dict,\
                 data_stream_queue_front_to_back, data_stream_queue_back_to_front):
        self.name       = instance_name
        self.frontend_ready_for_simulation = False

        # 1. configuration
        self.environment_type = environment_type
        self.brain_type       = brain_type
        self.has_reservoir    = has_reservoir
        self.learning_type    = learning_type

        # 2. state
        self.simulation_chunk_size = 1
        self.high_speed_mode       = False
        self.running               = True

        # 3. visualisation related state
        self.selected_by_frontend       = False

        # 4. main communications
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_backend, queue_to_backend, shared_dict, self.listeners)

        # 5. data stream communications
        self.data_stream_listeners = _Listeners()
        self.data_stream_comms     = Communications(data_stream_queue_front_to_back, data_stream_queue_back_to_front, shared_dict, self.data_stream_listeners)

        # 6. CUDA context (not yet entered: will only enter when the process starts)
        self.cuda_ctx = cuda_context

        # 7. adding diverse listeners
        self.build_listeners()

    def initialise(self):
        # 1. actually launch the CUDA context
        self.cuda_ctx.enter()
        if not self.cuda_ctx.check_yoself():
            raise RuntimeError(f"SandboxInstance '{self.environment_type} - {self.brain_type}': CUDA context is not valid!")
        # 2. initialise environment and brain
        """ self.environment = ...
        self.brain       = brain_spawner(self.brain_type, self.learning_type, self.cuda_ctx, self.name+"_brain", self.environment, self.has_reservoir)
        self.environment.attach_brain(self.brain) """
        
    def routine(self):
        import time
        # 1. initialise brain and environment
        self.initialise()

        # 2. send and wait for the frontend to process the instance specific
        self.comms.send("info for frontend", self._make_info_for_frontend())
        while not self.frontend_ready_for_simulation:
            time.sleep(0.1)
            self.process_messages()

        # 2. run
        EMA_simulated_seconds_per_real_second = 1.0
        while self.process_messages():
            # 1. run simulation chunk
            tic       = time.time()

            """ environment.run_n_steps(self.simulation_chunk_size, self.selected_by_frontend, skip_visuals=self.high_speed_mode, data_stream_comms=self.data_stream_comms) """
            time.sleep(np.random.uniform(0.01, 0.02) * self.simulation_chunk_size) 

            if self.selected_by_frontend:
                elapsed_s = time.time() - tic
                EMA_simulated_seconds_per_real_second = 0.95 * EMA_simulated_seconds_per_real_second + 0.05 * ((self.simulation_chunk_size * TIMESTEP_DURATION_MS / 1000.0) / elapsed_s)
                self.comms.send("seconds per real second update", {"sec per sec": EMA_simulated_seconds_per_real_second, "chunk size": self.simulation_chunk_size})
            # 2. sleep a bit to reduce the probability of overwriting the data stream towards the frontend (sleep is negiligible when fast mode)
            time.sleep(0.06) # the value should be adapted to the computer

    def build_listeners(self):
        self.add_listener(comms_prefix(self.name) + "set simulation chunk size", self._set_simulation_chunk_size)

    def exit_program(self, data):
        self.environment.streams.sync_all()
        self.data_stream_comms.empty_queues()
        self.comms.empty_queues()
        self.comms.cancel_join_threads()
        self.cuda_ctx.exit()
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
            name   = f"SandboxInstance Process: {self.environment_type} - {self.brain_type}",
            daemon = False,
        )
        process.start()
        return process

    """ def _make_info_for_frontend(self):
        info = {
                "instance name": self.name,
                "n_inp":    self.brain.n_inp,
                "n_out":    self.brain.n_out,
                "n_res":    self.brain.n_res,
                "n_noninp": self.brain.n_noninputs,
                "n_neurs":  self.brain.n_neurons,
            }
        self.environment.make_sandbox_specific_info(info)
        self.brain.make_sandbox_specific_info(info)
        return info """

    def _set_simulation_chunk_size(self, data):
        self.simulation_chunk_size = data
        self.high_speed_mode = self.simulation_chunk_size > 200
        self.environment._new_simulation_chunk_size(self.simulation_chunk_size, self.high_speed_mode)

    def _handle_frontend_ready(self, data):
        self.frontend_ready_for_simulation = True
