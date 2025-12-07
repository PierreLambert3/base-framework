import time

""" Shared dictionary wrapper """
class _Shared_dict:
    def __init__(self, shared_dict):
        self._dict = shared_dict
    
    def get(self, key, default=None):
        return self._dict.get(key, default)
    
    def set(self, key, value):
        self._dict[key] = value
    
    def remove(self, key):
        if key in self._dict:
            del self._dict[key]
    
    def keys(self):
        return self._dict.keys()
    
    def items(self):
        return self._dict.items()

""" Listeners: {"message name" : <function or method to call>} """
class _Listeners:
    def __init__(self):
        self.listeners = {}
    
    def add(self, message_name, callback):
        assert callable(callback), "add() : Callback must be callable."
        assert message_name not in self.listeners, f"Listener for '{message_name}' already exists."
        self.listeners[message_name] = callback
    
    def remove(self, message_name):
        assert message_name in self.listeners, f"Listener for '{message_name}' does not exist."
        del self.listeners[message_name]

    def __contains__(self, message_name):
        return message_name in self.listeners
    
    def __getitem__(self, message_name):
        return self.listeners[message_name]
    
""" Wraps a inter-process Queue """
class _Receiver:
    def __init__(self, queue_from_backend, listeners):
        self._queue = queue_from_backend
        self.listeners = listeners
    
    def process(self):
        while not self._queue.empty():
            event_name, event_data = self._queue.get()
            if event_name in self.listeners:
                self.listeners[event_name](event_data)
            else:
                print(f"Warning: unhandled event '{event_name}' received with data: {event_data}")


""" Wraps a inter-process Queue """
class _Sender:
    def __init__(self, queue_to_backend):
        self._queue = queue_to_backend
        self._send_timers = {} # event_name -> last send time (for throttling)
    
    def send(self, event_name, event_data=None, min_interval_s=0.1):
        if min_interval_s == 0.0:
            self._queue.put((event_name, event_data))
        else:
            last_send_time = self._send_timers.get(event_name, 0)
            now = time.time()
            if now - last_send_time >= min_interval_s:
                self._queue.put((event_name, event_data))
                self._send_timers[event_name] = now
