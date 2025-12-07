import multiprocessing
import time
from rendercanvas.auto import loop
import numpy as np

from GUI.engine.frontend.logic import Front_End
from GUI.pages.example_page import Page1

class Custom_Frontend(Front_End):
    def __init__(self, multiprocessing_context, queue_from_backend, queue_to_backend, window_name="Custom GUI Frontend Window"):
        super().__init__(queue_from_backend, queue_to_backend, window_name=window_name)
    
    def build_pages(self):
        self.pages['main'] = Page1(self.scene, "The Main Page")

    def build_listeners(self):
        pass

    def one_frame(self):
        self.fps_update()
        print("Frame Time EMA: {:.2f} ms".format(self.frame_time_EMA * 1000))

        # --- 1. logic ---
        scatterplot = self.pages['main'].get("My Scatterplot")
        scatterplot.receive_data(np.random.rand(scatterplot.n, 3).astype(np.float32) * 2 - 1)

        # --- 2. render ---
        self.scene.render()

        # --- 3. schedule next frame ---
        self.scene.canvas.request_draw(self.one_frame)

    def routine(self):
        # 1. initialisation
        self.initialise_scene()
        self.build_pages()
        self.build_listeners()

        # 2. Kickstart the loop
        self.scene.canvas.request_draw(self.one_frame)

        # 3. Blocking event loop: handles events and calls the custom one_frame().
        loop.run()

    def start(self):
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = "Frontend Process",
            daemon = False,
        )
        process.start()
        return process