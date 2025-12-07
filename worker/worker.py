from GUI.engine.backend.logic import Back_End

class Custom_Backend(Back_End):
    def __init__(self, multiprocessing_context, queue_from_frontend, queue_to_frontend):
        super().__init__(queue_from_frontend, queue_to_frontend)

    def routine(self):
        for i in range(10):
            print(f"Backend working... {i}")
            import time
            time.sleep(0.25)