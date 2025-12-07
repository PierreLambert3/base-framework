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

            # 1. receive things / read from shared dict
            self.process_messages()
            self.process_shared_dict()

            # 2. send things / write to shared dict
            self.send_messages()
            self.update_shared_variables()

    def send_messages(self):
        scatterplot_data = np.random.rand(1_000_000, 3).astype(np.float32) * 2 - 1
        self.send("scatterplot data", scatterplot_data)
    
    def update_shared_variables(self):
        pass

    def build_listeners(self):
        self.add_listener("frontend ping", self.on_frontend_ping)

    def on_frontend_ping(self, data):
        print(f"Backend received frontend ping with data: {data}")

    def process_shared_dict(self):
        self.mouse_position = self.read_shared_dict("pointer position", (0, 0))