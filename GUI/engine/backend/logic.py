from GUI.engine.comms import _Listeners, _Receiver, _Sender

class Back_End:
    def __init__(self, queue_from_frontend, queue_to_frontend):
        self.receiver  = _Receiver(queue_from_frontend) # from front end
        self.sender    = _Sender(queue_to_frontend)     # to front end
        self.listeners = _Listeners(self.receiver)
        self.listeners.add("exit program", self.exit_program)

    def routine(self):
        pass

    def exit_program(self, data):
        print("---  Back_End: exit_program received, this should be implemented in the derived class!  ---")
