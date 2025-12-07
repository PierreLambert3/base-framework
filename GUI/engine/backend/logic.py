from GUI.engine.comms import _Listeners, _Receiver, _Shared_dict, _Sender

class Back_End:
    def __init__(self, queue_from_frontend, queue_to_frontend, shared_dict):
        self.listeners = _Listeners()
        self.receiver  = _Receiver(queue_from_frontend, self.listeners) # from front end
        self.sender    = _Sender(queue_to_frontend)     # to front end
        self.shared    = _Shared_dict(shared_dict)
        self.add_listener("exit program", self.exit_program)

    def routine(self):
        pass

    def build_listeners(self):
        raise Exception("--- Back_End.build_listeners() should be implemented in the derived class! ---")

    def exit_program(self, data):
        print("---  Back_End: exit_program received, this should be implemented in the derived class!  ---")

    def send(self, event_name, event_data=None, min_interval_s=0.0):
        self.sender.send(event_name, event_data, min_interval_s)

    def process_messages(self):
        self.receiver.process()

    def update_shared_dict(self, key, value):
        self.shared.set(key, value)

    def process_shared_dict(self):
        raise Exception("--- Back_End.process_shared_dict() should be implemented in the derived class! ---")

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def read_shared_dict(self, key, value):
        self.shared.set(key, value)