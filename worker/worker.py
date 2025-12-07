import numpy as np
from GUI.engine.backend.logic import Back_End


class Custom_Backend(Back_End):
    def __init__(self, multiprocessing_context, queue_from_frontend, queue_to_frontend, shared_dict):
        super().__init__(queue_from_frontend, queue_to_frontend, shared_dict)

    def routine(self):
        self.build_listeners()
        scatterplot_data = np.random.rand(1_000_000, 3).astype(np.float32) * 2 - 1
        for i in range(100):
            import time
            time.sleep(0.05)

            self.process_messages()    # from Queue (careful not to saturate the queue)
            self.process_shared_dict() # from shared dict
            self.send_messages()       # to Queue
            self.update_shared_variables() # to shared dict


    def send_messages(self):
        self.send("backend ping", "Ping from backend", min_interval_s=0.5)
    
    def update_shared_variables(self):
        scatterplot_data = np.random.rand(1_000_000, 3).astype(np.float32) * 2 - 1
        self.update_shared_dict("scatterplot data", scatterplot_data)

    def build_listeners(self):
        self.add_listener("frontend ping", self.on_frontend_ping)

    def on_frontend_ping(self, data):
        print(f"Backend received frontend ping with data: {data}")

    def process_shared_dict(self):
        self.mouse_position = self.shared.get("pointer position", (0, 0))