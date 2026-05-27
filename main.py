# Wiki: see wiki/01-architecture.md (process layout) and
# wiki/09-example-walkthrough.md (end-to-end trace).
# This is the entry point: it creates the multiprocessing context, the
# inter-process queues and the shared dict, then starts the frontend
# (separate process) and the backend (either blocking in this process,
# or as a separate process when ASYNC_BACKEND = True).

import multiprocessing
import time

from GUI.gui              import Custom_Frontend
from GUI.backend          import Custom_Backend
from GUI.engine.main_side import Main_Side


ASYNC_BACKEND = False
# ASYNC_BACKEND = True


def drain_queue(q):
    try:
        while not q.empty():
            q.get_nowait()
    except:
        pass


if __name__ == "__main__":

    # 1. initialise the multiprocessing context
    ctx = multiprocessing.get_context("spawn")
    # ctx.freeze_support()

    # 2. create the communication queues
    front_to_back_queue = ctx.Queue()
    back_to_front_queue = ctx.Queue()
    manager             = ctx.Manager()  # Keep reference to shutdown later
    shared_dict         = manager.dict() # for shared that shouldn't saturate the queues (e.g. mouse position updates)

    # 3. create & launch the front-end process (separate process)
    frontend = Custom_Frontend(ctx, back_to_front_queue, front_to_back_queue, shared_dict, window_name="my custom window")
    frontend_process = frontend.start()

    # 4. create & run the back-end (blocking in this process, OR as its own process)
    if not ASYNC_BACKEND:
        # ---- blocking mode ----
        backend = Custom_Backend(ctx, manager, front_to_back_queue, back_to_front_queue, shared_dict)
        backend.routine()  # <-- this is a blocking call
    else:
        # ---- async mode ----
        main_to_back_queue = ctx.Queue()
        back_to_main_queue = ctx.Queue()

        backend = Custom_Backend(
            ctx, manager, front_to_back_queue, back_to_front_queue, shared_dict,
            queue_from_main=main_to_back_queue, queue_to_main=back_to_main_queue,
        )
        backend_process = backend.start()

        main_side = Main_Side(back_to_main_queue, main_to_back_queue, shared_dict)

        # --- register any main->backend shared arrays here ---
        # 1. Mallocs and registers a CPU side array with main_side.register_shared_array_for_backend
        # arr = main_side.register_shared_array_for_backend("foo", (1024,), np.float32)
        # 2. Write to "arr" periodically from main process and send an event to let it know
        #    backend reads via self.get_main_shared_array("foo") 
        #    and listens for "main:..." events sent using main_side.send(...)

        # main-process loop: poll comms, watch backend liveness
        while main_side.running and backend_process.is_alive():
            print("main process doing its thing...")
            time.sleep(0.2)
            main_side.process_messages()

        # signal backend to exit (no-op if it already did), then wait
        main_side.send("exit program", None, needs_ack=False)
        backend_process.join(timeout=5.0)
        if backend_process.is_alive():
            print("WARNING: Backend is slow to exit. Exiting in 30 seconds if it does not respond until then.")
            backend_process.join(timeout=30.0)
            print("--- Backend process did not exit cleanly, terminating... ---")
            backend_process.terminate()
            backend_process.join()

        # release shared-memory blocks owned by the main side
        main_side.comms.release_all_shared_arrays()
        drain_queue(main_to_back_queue)
        drain_queue(back_to_main_queue)

    # 5. closure: tell frontend to exit and wait for it
    back_to_front_queue.put(("exit program", None))
    front_to_back_queue.put(("exit program", None))

    # 5.1 wait for frontend process
    frontend_process.join(timeout=2.0)
    if frontend_process.is_alive():
        print("--- Frontend process did not exit cleanly, terminating... ---")
        frontend_process.terminate()
        frontend_process.join()
    # 5.2 clear queues
    drain_queue(front_to_back_queue)
    drain_queue(back_to_front_queue)