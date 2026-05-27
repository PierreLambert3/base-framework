# Wiki: wiki/01-architecture.md, wiki/02-communications.md
# Engine helper for the *main* process when the backend is launched as a
# child process (async backend mode, see main.py).
#
# Mirrors the (frontend <-> backend) pattern: a dedicated pair of queues
# wired through a `Communications` instance, with `send()` + listener
# registration via `add_listener(...)`. Also exposes a helper to allocate
# shared-memory numpy arrays that the backend can attach to, tracked in
# `self.shared_arrays` (initially empty).
#
# Event names sent from main must not collide with frontend-originated
# event names: the backend shares one `_Listeners` registry between its
# frontend-facing and main-facing `Communications`.
#
# When the backend runs blocking inside the main process (default), this
# module is unused.

from GUI.engine.comms import _Listeners, Communications
from GUI.engine.backend.logic import REGISTER_SHARED_ARRAY_FROM_MAIN, REGISTER_SHARED_ARRAY_FOR_WORKER
from GUI.engine.shared_array import SharedArray


class Main_Side:
    def __init__(self, queue_from_backend, queue_to_backend, shared_dict):
        self.listeners = _Listeners()
        self.comms     = Communications(queue_from_backend, queue_to_backend, shared_dict, self.listeners)
        self.running   = True

        # Tracks the shared-memory arrays this side has allocated for the
        # backend to attach to. Initially empty; populated by
        # `register_shared_array_for_backend(...)`.
        self.shared_arrays = {} # key -> SharedArray

        # Default: a backend-initiated shutdown also stops the main loop.
        self.add_listener("exit program", self.exit_program)

    # ------------------------------------------------------------- messaging
    def add_listener(self, event_name, callback):
        self.listeners.add(event_name, callback)

    def send(self, event_name, event_data=None, needs_ack=True):
        self.comms.send(event_name, event_data, needs_ack=needs_ack)

    def process_messages(self):
        self.comms.process_messages()

    # ----------------------------------------------------------- shared dict
    def update_shared_dict(self, key, value):
        self.comms.shared.set(key, value)

    def read_shared_dict(self, key, default=None):
        return self.comms.shared.get(key, default=default)

    # ----------------------------------------------- shared-memory arrays
    def register_shared_array_for_backend(self, key, shape, dtype):
        """Allocate a shared-memory numpy array on the main side and ship the
        descriptor to the backend. Returns a `SharedArray` wrapper. Subsequent
        growth is handled by the wrapper itself (auto-broadcasts the new
        descriptor via the `on_reallocated` callback)."""
        if key in self.shared_arrays:
            return self.shared_arrays[key]
        def _on_reallocated(info, _key=key):
            self.send(REGISTER_SHARED_ARRAY_FROM_MAIN, {"key": _key, **info}, needs_ack=True)
        sa = SharedArray(self.comms, key, shape, dtype, on_reallocated=_on_reallocated)
        self.shared_arrays[key] = sa
        info = self.comms.get_shared_array_info(key)
        self.send(REGISTER_SHARED_ARRAY_FROM_MAIN, {"key": key, **info}, needs_ack=True)
        return sa

    def get_shared_array(self, key):
        return self.shared_arrays.get(key)

    # ----------------------- shared-memory arrays for a worker instance
    def register_shared_array_for_worker(self, instance_name, key, shape, dtype):
        """Allocate a shared-memory numpy array on the main side and ship its
        descriptor to the backend, which (a) attaches it on the backend side
        and (b) forwards the descriptor to the named worker. Returns a
        `SharedArray` wrapper; subsequent growth is handled by the wrapper
        (auto-broadcasts the new descriptor via `on_reallocated`)."""
        if key in self.shared_arrays:
            return self.shared_arrays[key]
        def _on_reallocated(info, _name=instance_name, _key=key):
            self.send(REGISTER_SHARED_ARRAY_FOR_WORKER,
                      {"instance_name": _name, "key": _key, **info},
                      needs_ack=True)
        sa = SharedArray(self.comms, key, shape, dtype, on_reallocated=_on_reallocated)
        self.shared_arrays[key] = sa
        info = self.comms.get_shared_array_info(key)
        self.send(REGISTER_SHARED_ARRAY_FOR_WORKER,
                  {"instance_name": instance_name, "key": key, **info},
                  needs_ack=True)
        return sa

    # ---------------------------------------------------------------- exit
    def exit_program(self, data=None):
        if not self.running:
            return
        self.running = False
        # Notify the backend
        self.comms.send("exit program", None, needs_ack=False)
        self.comms.cancel_join_threads()
