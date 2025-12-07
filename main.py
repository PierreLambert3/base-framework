import multiprocessing

from GUI.gui       import Custom_Frontend
from worker.worker import Custom_Backend


if __name__ == "__main__":

    # 1. initialise the multiprocessing context
    ctx = multiprocessing.get_context("spawn")
    ctx.freeze_support()

    # 2. create the communication queues
    front_to_back_queue = ctx.Queue()
    back_to_front_queue = ctx.Queue()

    # 3. create & launch the front-end process (separate process)
    frontend = Custom_Frontend(ctx, back_to_front_queue, front_to_back_queue, window_name="my custom window")
    frontend_process = frontend.start()

    # 4. create & launch the back-end process (this process)
    backend  = Custom_Backend(ctx, front_to_back_queue, back_to_front_queue)
    backend.routine()  # <-- this is a blocking call

    # 5. at this point: 
    #       -  backend process is done
    #       -  front-end process might still be running: signal it to finish & wait for it
    back_to_front_queue.put(("exit program", None))
    frontend_process.join()