# Wiki: wiki/02-communications.md
# Centralised inter-process communication: event queues with automatic ACK
# (latest-wins flow control) and a Manager().dict() for hot continuous state.
# Read alongside the wiki page; this is the single source of truth for the
# event/ACK protocol used by frontend, backend, and worker instances.

import time
from multiprocessing import shared_memory
import numpy as np

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
    def __init__(self, queue_rcv, listeners):
        self._queue = queue_rcv
        self.listeners = listeners
    
    def process(self, sender):
        ack_received = False
        while not self._queue.empty():
            # 1. process incoming message
            message = self._queue.get()
            if len(message) == 2:
                event_name, event_data = message
                needs_ack = True
            else:
                event_name, event_data, needs_ack = message
            
            if event_name in self.listeners:
                self.listeners[event_name](event_data)
            else:
                print(f"Warning: unhandled event '{event_name}' received with data: {event_data}")
            # 2. send ack if not an ack event and if the sender requested an ack
            is_ack_event = event_name.endswith(" ack")
            if needs_ack and not is_ack_event:
                sender.send(event_name + " ack", event_name)
            ack_received = ack_received or is_ack_event
        return ack_received
    
    def send_to_self(self, event_name, event_data=None):
        self._queue.put((event_name, event_data))

""" Wraps a inter-process Queue """
class _Sender:
    def __init__(self, queue_out):
        self._queue = queue_out
        self._send_timers = {} # event_name -> last send time (for throttling)
    
    def send(self, event_name, event_data=None, needs_ack=True):
        self._queue.put((event_name, event_data, needs_ack))

class Communications:
    def __init__(self, queue_rcv, queue_out, shared_dict, listeners):

        # - queues: event based communication, with automatic ack handling
        self.listeners = listeners
        self.receiver  = _Receiver(queue_rcv, self.listeners)
        self.sender    = _Sender(queue_out)
        self.outgoing_messages_ready = {} # [event_name] = ready to send   (waits for ack of previous message)
        self.pending_outgoing        = {} # if not acked yet: put here the data to send later
        self.pending_no_ack          = {} # for send_without_ack: latest data per event_name (overrides)

        # - for continuous data sharing (not event based): shared dictionary
        self.shared    = _Shared_dict(shared_dict) 

        # - shared-memory arrays for high-throughput numpy data (zero-copy
        #   inter-process transfer; used as an alternative to shipping large
        #   arrays through the queues). Tracked here so they can be released
        #   on teardown. See `create_shared_array`/`attach_shared_array`.
        self._owned_shm    = {} # key -> (shm, ndarray, shape, dtype)
        self._attached_shm = {} # key -> (shm, ndarray, shape, dtype)

    def _register_outgoing_message(self, event_name):
        assert event_name not in self.outgoing_messages_ready, f"Outgoing message '{event_name}' already registered."
        self.outgoing_messages_ready[event_name] = True
        self.receiver.listeners.add(event_name + " ack", self._on_outgoing_message_ack)

    def _on_outgoing_message_ack(self, data):
        event_name = data
        self.outgoing_messages_ready[event_name] = True
    
    def _send_pending(self):
        for event_name in list(self.pending_outgoing.keys()):
            if self.outgoing_messages_ready.get(event_name):
                data = self.pending_outgoing[event_name]
                self.send(event_name, data)
                del self.pending_outgoing[event_name]
                self.outgoing_messages_ready[event_name] = False

    def send(self, event_name, event_data=None, needs_ack=True):
        # If no ack needed, bypass flow control entirely
        if not needs_ack:
            self.sender.send(event_name, event_data, needs_ack=False)
            return
        # Register the message if not already done
        if event_name not in self.outgoing_messages_ready:
            self._register_outgoing_message(event_name)
        # Send or set as pending
        has_overridden = False
        if self.outgoing_messages_ready[event_name]:
            self.sender.send(event_name, event_data, needs_ack=True)
            self.outgoing_messages_ready[event_name] = False
        else:
            self.pending_outgoing[event_name] = event_data
            has_overridden = True
        return has_overridden

    def process_messages(self):
        ack_received = self.receiver.process(self.sender)
        if ack_received:
            self._send_pending()

    def empty_queues(self):
        try:
            while not self.sender._queue.empty():
                self.sender._queue.get_nowait()
        except:
            pass
        try:
            while not self.receiver._queue.empty():
                self.receiver._queue.get_nowait()
        except:
            pass

    def cancel_join_threads(self):
        """Call before process exit to prevent hanging on queue feeder threads.
        This tells Python not to wait for background threads that flush data to pipes.
        Manager-backed queues (AutoProxy[Queue]) don't expose this method, so we skip them."""
        for q in (self.sender._queue, self.receiver._queue):
            cancel = getattr(q, "cancel_join_thread", None)
            if callable(cancel):
                cancel()

    # -------------------------------------------- shared-memory numpy arrays
    def create_shared_array(self, key, shape, dtype):
        """Create a new shared-memory block big enough to hold a numpy array
        of the given shape and dtype. Returns a numpy view onto that block.

        The owning process must keep the returned `Communications` alive for
        the lifetime of the array (we hold a reference to the `SharedMemory`
        handle in `_owned_shm`). Call `release_shared_array(key)` or
        `release_all_shared_arrays()` to free it.
        """
        assert key not in self._owned_shm,    f"Shared array '{key}' already owned by this Communications."
        assert key not in self._attached_shm, f"Shared array '{key}' already attached to this Communications."
        dtype = np.dtype(dtype)
        n_bytes = int(np.prod(shape)) * dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=n_bytes)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self._owned_shm[key] = (shm, arr, tuple(shape), dtype)
        return arr

    def attach_shared_array(self, key, name, shape, dtype):
        """Attach to an existing shared-memory block by `name` (created by
        another process via `create_shared_array`). Returns a numpy view.
        Call `release_shared_array(key)` or `release_all_shared_arrays()` to
        detach (only the owning side should `unlink`)."""
        assert key not in self._owned_shm,    f"Shared array '{key}' already owned by this Communications."
        assert key not in self._attached_shm, f"Shared array '{key}' already attached to this Communications."
        dtype = np.dtype(dtype)
        shm = shared_memory.SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        self._attached_shm[key] = (shm, arr, tuple(shape), dtype)
        return arr

    def get_shared_array(self, key):
        """Return the numpy view for a previously created or attached shared array."""
        if key in self._owned_shm:
            return self._owned_shm[key][1]
        if key in self._attached_shm:
            return self._attached_shm[key][1]
        return None

    def get_shared_array_info(self, key):
        """Return `{"name", "shape", "dtype"}` for a previously *created* (owned)
        shared array. Use this to propagate the descriptor to other processes
        (e.g. via the `info for frontend` channel)."""
        assert key in self._owned_shm, f"Shared array '{key}' is not owned by this Communications (cannot describe an attached array)."
        shm, _arr, shape, dtype = self._owned_shm[key]
        return {"name": shm.name, "shape": list(shape), "dtype": dtype.str.lstrip("<>=|")}

    def release_shared_array(self, key):
        """Release a single shared array. Owning side closes + unlinks; the
        attached side only closes."""
        if key in self._owned_shm:
            shm, arr, _shape, _dtype = self._owned_shm.pop(key)
            del arr  # drop numpy view referencing the buffer
            try: shm.close()
            except Exception: pass
            try: shm.unlink()
            except Exception: pass
        elif key in self._attached_shm:
            shm, arr, _shape, _dtype = self._attached_shm.pop(key)
            del arr
            try: shm.close()
            except Exception: pass

    def release_all_shared_arrays(self):
        for key in list(self._attached_shm.keys()):
            self.release_shared_array(key)
        for key in list(self._owned_shm.keys()):
            self.release_shared_array(key)