from GUI.engine.comms import _Listeners, Communications

class Back_End:
    def __init__(self, queue_from_frontend, queue_to_frontend, shared_dict):
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_frontend, queue_to_frontend, shared_dict, self.listeners)
        self.add_listener("exit program", self.exit_program)

    def routine(self):
        pass

    def build_listeners(self):
        raise Exception("--- Back_End.build_listeners() should be implemented in the derived class! ---")

    def exit_program(self, data):
        print("---  Back_End: exit_program received, this should be implemented in the derived class!  ---")

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