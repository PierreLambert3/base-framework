# 2. Communication protocols

All inter-process communication is centralised in
[GUI/engine/comms.py](../GUI/engine/comms.py). Read the source alongside this
page; it is short and self-contained.

There are **three** communication mechanisms in this framework:

1. **Event queues with automatic ACK** — for control / requests / lifecycle.
2. **Data stream queues without ACK** — for high-throughput payloads
   (simulation results to draw).
3. **Shared dict (`Manager().dict()`)** — for tiny continuous state that
   shouldn't saturate a queue (e.g. live mouse position).

## 2.1 The `Communications` object

`Communications` wraps **one pair of queues** plus a reference to the shared
dict and a `_Listeners` registry. Every process / channel that talks to
another one instantiates one `Communications`:

```python
from GUI.engine.comms import _Listeners, Communications
listeners = _Listeners()
comms = Communications(queue_in, queue_out, shared_dict, listeners)
```

Three things you do with it:

| Action | Method |
|---|---|
| Subscribe to an event | `listeners.add("event name", callback)` |
| Send an event | `comms.send("event name", data, needs_ack=True)` |
| Drain incoming events (call every frame / tick) | `comms.process_messages()` |

Per-frame, a process **must** call `comms.process_messages()` on every
`Communications` it owns. The frontend does this in `one_frame()`, the
backend in its `routine()` loop, and each worker in `process_messages()`.

## 2.2 Event format

A message is a tuple `(event_name, event_data)` or
`(event_name, event_data, needs_ack)` (default `needs_ack=True`).

* `event_name` is a string. **It is your routing key.** Listeners are
  registered against this exact string.
* `event_data` is anything picklable (dicts, numpy arrays, …). Big numpy
  payloads are fine but cost a memcpy on each side of the queue.

## 2.3 The ACK / flow-control protocol

This is the most subtle part of the framework. Read carefully.

For events with `needs_ack=True` (the default):

1. Sender calls `comms.send("foo", data)`.
2. The receiver gets it, runs the listener, then **automatically** sends
   back `("foo ack", "foo")` (this is done by `_Receiver.process`).
3. Sender's `Communications` listens for `"foo ack"` and marks `"foo"`
   as **ready to send again**.
4. If the sender calls `send("foo", new_data)` while `"foo"` has not been
   ACKed yet, the new data **does not queue up** — it overrides the
   pending value:
   ```python
   self.pending_outgoing[event_name] = event_data   # latest wins
   ```
   Once the ACK arrives, the *latest* pending value is flushed.
   `send()` returns `True` when it had to override an unsent message.

This is **latest-wins flow control per event_name**. The queue can never
back up with redundant copies of the same event.

### Implications

* Receivers should be cheap (or run async work themselves) — ACKs are sent
  synchronously after the listener returns.
* Two listeners for the *same* `event_name` is forbidden:
  `_Listeners.add` asserts uniqueness.
* If you genuinely want to send rapid-fire copies that must all be
  delivered, set `needs_ack=False` (see §2.5).

### Naming convention used in the codebase

* Plain events: `"new worker instance created"`, `"exit program"`, …
* Question / response style: `"Q1: how many timesteps per simulation chunk"`
  → reply `"RE1.1: how many timesteps per simulation chunk"`.
* Backend → instance routing uses a name prefix:
  ```python
  comms_prefix(name) = "<" + name + "> "
  ```
  All listeners on the worker side are registered with that prefix so the
  backend's outbound `Communications` (shared with all instances) can route
  by `event_name`. See [worker/worker_instance.py](../worker/worker_instance.py).

## 2.4 The shared dict

`shared_dict = manager.dict()` is created in `main.py` and passed to every
`Communications`. It is exposed as `comms.shared`:

```python
comms.shared.set("mouse_xy", (x, y))
xy = comms.shared.get("mouse_xy", default=(0, 0))
```

Use it for:

* state that changes every frame (mouse, camera) and that the other side
  only needs to "peek" at,
* a value that you'd otherwise spam a queue with.

Do **not** use it for events: there's no notification, the consumer has to
poll.

## 2.5 The data stream channel (no ACK)

Worker instances stream simulation output to the frontend on **separate**
queues, with `needs_ack=False`. The backend creates the queue pair when it
spawns the worker:

```python
data_stream_front_to_back = self.manager.Queue()
data_stream_back_to_front = self.manager.Queue()
```

These queues are routed straight from worker → frontend; the backend
**doesn't** read them. The frontend creates a dedicated `Communications`
per worker instance (`data_stream_comms_per_instance`) so each instance has
its own listener namespace.

Worker side (see [worker/custom_worker.py](../worker/custom_worker.py)):

```python
self.data_stream_comms.send(
    "data stream: positions",
    {"positions": self._positions_host},
    needs_ack=False,            # << critical: bypass ACK flow control
)
```

Frontend side (see `Main_Page.on_new_worker_instance`):

```python
self.frontend.add_data_stream_listener(
    instance_name,
    "data stream: positions",
    lambda data: self._handle_positions_data(instance_name, data),
)
```

Because there is no ACK, **the queue can grow** if the frontend can't keep
up. In practice we drain the queue every frame and only the latest payload
matters visually, so this is acceptable. If you need stricter control, send
`needs_ack=True` events on the *main* channel instead.

## 2.6 Cheat sheet

```text
                 Queue/dict       ACK?   Use for
control event    FE↔BE / BE↔W      yes   "spawn this", "configure that"
data stream      W→FE              no    "here are 1M points to render"
shared dict      everyone          n/a   live mouse, hot scalars
```

## 2.7 Pitfalls

* **Forgetting to call `process_messages()`** every frame — your listeners
  will never fire and the queues will fill up.
* **Two listeners for the same name** — `_Listeners.add` will assert.
* **Heavy listener** that does GPU work synchronously on the frontend —
  blocks rendering. Push the work to a worker instead.
* **Sending huge unique payloads with `needs_ack=False`** — fine for
  dropable data, dangerous for must-deliver state.
