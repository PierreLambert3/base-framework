import multiprocessing
from GUI.gui       import Custom_Frontend
from worker.worker import Custom_Backend


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

    # 4. create & launch the back-end process (this process)
    backend  = Custom_Backend(ctx, manager, front_to_back_queue, back_to_front_queue, shared_dict)
    backend.routine()  # <-- this is a blocking call
    back_to_front_queue.put(("exit program", None))
    front_to_back_queue.put(("exit program", None))

    # 5.1 wait for frontend process
    frontend_process.join(timeout=2.0)
    if frontend_process.is_alive():
        print("--- Frontend process did not exit cleanly, terminating... ---")
        frontend_process.terminate()
        frontend_process.join()
    # 5.2 clear queues
    def drain_queue(q):
        try:
            while not q.empty():
                q.get_nowait()
        except:
            pass
    drain_queue(front_to_back_queue)
    drain_queue(back_to_front_queue)