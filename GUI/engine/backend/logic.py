# Wiki: wiki/03-backend.md (engine base class for the backend process).
# Engine file -- do not edit for project work; subclass `Back_End`
# (see `GUI/backend.py::Custom_Backend`).

import time
from GUI.engine.comms import _Listeners, Communications

DEAD_AFTER_S = 5.0

class Back_End:
    def __init__(self, multiprocessing_context, manager, queue_from_frontend, queue_to_frontend, shared_dict):
        self.multiprocessing_context = multiprocessing_context
        self.manager   = manager  # for creating queues that can be sent between processes
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_frontend, queue_to_frontend, shared_dict, self.listeners)
        self.running   = True

    def routine(self):
        pass

    def build_listeners(self):
        raise Exception("--- Back_End.build_listeners() should be implemented in the derived class! ---")

    def exit_program(self, data):
        self.running = False
        self.comms.send("exit program", None, needs_ack=False)
        self.comms.cancel_join_threads()

    def send(self, event_name, event_data=None, min_interval_s=0.0):
        self.comms.send(event_name, event_data)

    def process_messages(self):
        self.comms.process_messages()

    def update_shared_dict(self, key, value):
        self.comms.shared.set(key, value)

    def process_shared_dict(self):
        raise Exception("--- Back_End.process_shared_dict() should be implemented in the derived class! ---")

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def read_shared_dict(self, key, default=None):
        return self.comms.shared.get(key, default=default)
    
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