import numpy as np
import time 

from GUI.engine.backend.logic import Back_End
from GUI.engine.comms import Communications, _Listeners
from worker.sandbox_instance import SandboxInstance, comms_prefix


class Custom_Backend(Back_End):
    def __init__(self, multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict):
        super().__init__(multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict)
        self.add_listener("exit program", self.exit_program)

        # worker processes that will be spawned by the backend
        self.sandbox_instance_names = []
        self.sandbox_processes = []
        self.comms_instances = []
        self.cuda_contexts = []
        self.data_stream_queues = []  # list of tuples: (from_backend, to_backend) from frontend's perspective
        self.n_instances_created = 0

    def routine(self):
        self.build_listeners()
        # 1. fetch initial information from frontend
        self.send("Q1: how many timesteps per simulation chunk", None)
        # 2. main loop
        while self.running:
            time.sleep(0.25) # no need to be too reactive here
            self.process_messages()

    def process_messages(self):
        super().process_messages()
        # process instance queues as well (ACKs + any instance-originated events).
        for comms in list(self.comms_instances):
            comms.process_messages()

    def build_listeners(self):
        self.add_listener("RE1.1: how many timesteps per simulation chunk", self._set_simulation_chunk_size)

    def _set_simulation_chunk_size(self, data):
        self.simulation_chunk_size = data

    def exit_program(self, data):
        super().exit_program(data)
        for i in range(len(self.sandbox_instance_names)):
            self.send_to_instance(self.sandbox_instance_names[i], "exit program", None, require_ack=False)
        for i, proc in enumerate(self.sandbox_processes):
            proc.join()
        
    def _handle_launch_sandbox_pair(self, data):
        self.n_instances_created += 1
        environment_type = data["environment"]
        brain_type       = data["brain"]
        has_reservoir    = data["has_reservoir"]
        learning_type    = data["learning type"]

        # 1.1 Communications: backend <-> subprocess
        backend_to_subprocess = self.multiprocessing_context.Queue()
        subprocess_to_backend = self.multiprocessing_context.Queue()
        comms_to_instance     = Communications(subprocess_to_backend, backend_to_subprocess, self.comms.shared, self.listeners)
        # 1.2 Data stream queues: subprocess --> frontend
        data_stream_front_to_back = self.manager.Queue()
        data_stream_back_to_front = self.manager.Queue()
        
        # 2. CUDA context
        cuda_ctx = self.cuda_manager.create_context()
        
        # 3. start the process & save references
        instance_name = f"SandboxInstance: {environment_type} - {brain_type} - {learning_type}  " + str(self.n_instances_created)
        instance      = SandboxInstance(cuda_ctx, instance_name, environment_type, brain_type, has_reservoir, learning_type, subprocess_to_backend, backend_to_subprocess, self.comms.shared, data_stream_front_to_back, data_stream_back_to_front)
        process       = instance.start()
        self.sandbox_processes.append(process)
        self.sandbox_instance_names.append(instance_name)
        self.comms_instances.append(comms_to_instance)
        self.cuda_contexts.append(cuda_ctx)
        self.data_stream_queues.append((data_stream_front_to_back, data_stream_back_to_front))

        # 4. inform the instance (and all others) of the simulation chunk size
        self._set_simulation_chunk_size(self.simulation_chunk_size)

        # 5. inform the frontend that a new instance has been created, including the dedicated data stream queues
        self.send("SandboxPage: new sandbox instance created", {
            "instance name": instance_name,
            "environment": environment_type,
            "brain": brain_type,
            "has_reservoir": has_reservoir,
            "learning type": learning_type,
            "data_stream_queues": (data_stream_back_to_front, data_stream_front_to_back)  # (from_backend, to_backend) from frontend's perspective
        }) 
   
        
    
    def send_to_instance(self, instance_name, event_name, event_data=None, require_ack=True):
        failure= True
        for i in range(len(self.sandbox_instance_names)):
            if self.sandbox_instance_names[i] == instance_name:
                self.comms_instances[i].send(comms_prefix(instance_name) + event_name, event_data)
                failure = False
        if failure:
            print(f"FAILURE: could not send to instance '{instance_name}' because it does not exist!")
   

    
    def _handle_instance_deselected(self, data): # data == instance_name
        self.send_to_instance(data, "sandbox instance deselected", None)
   