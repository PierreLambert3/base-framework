import time

""" Listeners: {"message name" : <function or method to call>} """
class _Listeners:
    def __init__(self, receiver):
        self.listeners = {}
        self.receiver = receiver
    
    def add(self, message_name, callback):
        assert callable(callback), "add() : Callback must be callable."
        assert message_name not in self.listeners, f"Listener for '{message_name}' already exists."
        self.listeners[message_name] = callback
    
    def remove(self, message_name):
        assert message_name in self.listeners, f"Listener for '{message_name}' does not exist."
        del self.listeners[message_name]
    
    def __call__(self):
        while not self.receiver._queue.empty():
            message_name, message_data = self.receiver._queue.get()
            assert message_name in self.listeners, f"No listener for message '{message_name}'."
            self.listeners[message_name](message_data)

""" Wraps a inter-process Queue """
class _Receiver:
    def __init__(self, queue_from_backend):
        self._queue = queue_from_backend
        self.events = {} # event_name -> function or method to call when event received
    
    def process(self):
        while not self._queue.empty():
            event_name, event_data = self._queue.get()
            if event_name in self.events:
                self.events[event_name](event_data)
            else:
                print(f"Warning: unhandled event '{event_name}' received with data: {event_data}")



""" Wraps a inter-process Queue """
class _Sender:
    def __init__(self, queue_to_backend):
        self._queue = queue_to_backend
        self._send_timers = {} # event_name -> last send time (for throttling)
    

    def send(self, event_name, event_data=None, min_interval_s=0.0):
        # 1. verify that this particular event is not already in the queue
        1/0

        # 2. send event (possibly skip if throttling in effect)
        if min_interval_s == 0.0:
            self._queue.put((event_name, event_data))
        else:
            last_send_time = self._send_timers.get(event_name, 0)
            now = time.time()
            if now - last_send_time >= min_interval_s:
                self._queue.put((event_name, event_data))
                self._send_timers[event_name] = now
