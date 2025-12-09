import multiprocessing
import time
from rendercanvas.auto import loop
import numpy as np

from GUI.engine.frontend.logic import Front_End
from GUI.pages.example_page import Page1

TESTING_PERFORMANCE = False

class Custom_Frontend(Front_End):
    def __init__(self, multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name="Custom GUI Frontend Window"):
        super().__init__(multiprocessing_context, queue_from_backend, queue_to_backend, shared_dict, window_name=window_name)
        self.set_fps(10)

    def build_pages(self):
        self.add_page("main", Page1(self.scene, "The Main Page"))

    def build_listeners(self):
        self.add_listener("scatterplot data", self.on_scatterplot_data)

    def on_user_event(self, event):
        mouse_event = (event["event_type"] == "pointer_move" or event["event_type"] == "pointer_down" or event["event_type"] == "pointer_up")
        
        if mouse_event:
            # 1. update pointer position in shared dict (raw screen coords)
            screen_mouse_coords = (event["x"], event["y"])
            if event["event_type"] == "pointer_move":
                self.update_shared_dict("pointer position", (screen_mouse_coords[0], screen_mouse_coords[1]))

            # 2. check for hits on current page & get page-relative coords
            if self.current_page is not None:
                page_coords = self.scene.xy_on_mesh(screen_mouse_coords, self.current_page.pick_mesh)
                if page_coords is not None:
                    if event["event_type"] == "pointer_move":
                        self.manage_mouse_pointer_move_in_page(event, self.current_page, page_coords)
                    elif event["event_type"] == "pointer_down":
                        self.manage_mouse_pointer_down_in_page(event, self.current_page, page_coords)
                    elif event["event_type"] == "pointer_up":
                        self.manage_mouse_pointer_up_in_page(event, self.current_page, page_coords)
            

    def process_shared_dict(self):
        pass

    def one_frame(self):
        if not self.should_it_render():
            self.scene.canvas.request_draw(self.one_frame)
            return

        # --- 1. render ---
        self.scene.render()

        # --- 2. schedule next frame ---
        self.scene.canvas.request_draw(self.one_frame)

        # --- 3. logic ---
        if TESTING_PERFORMANCE:
            scatterplot = self.pages['main'].get("My Scatterplot")
            scatterplot.receive_data(np.random.rand(scatterplot.n, 3).astype(np.float32) * 2 - 1)
        else:
            self.process_messages()    # from Queue (careful not to saturate the queue)
            self.process_shared_dict() # from shared dict
        # rotate elements for demo purposes
        parallelepiped = self.pages['main'].get("My Parallelepiped")
        parallelepiped.rotate((0.15, 0.05, 0.05))  # degrees per frame
        inner_container = self.pages['main'].get("Inner Container")
        inner_container.rotate((0.01, 0.05, 0.0))
        # change bloom tint over time, just for the demo
        if self.scene.bloom is not None:
            # Animate bloom tint color over time
            t = time.time()
            t *= 2.5
            bloom_tint = np.array([
                0.3 + 0.7 * np.sin(t * 0.5),
                0.3 + 0.7 * np.sin(t * 0.7 + 2.0),
                0.3 + 0.7 * np.sin(t * 0.9 + 4.0),
            ], dtype=np.float32)
            # self.scene.bloom._uniform_data["color_center"] = (np.random.uniform(), np.random.uniform(), np.random.uniform())
            self.scene.bloom.set_params(color_center=bloom_tint)

    def routine(self):
        # 1. initialisation
        try:
            self.initialise_scene()
            self.build_pages()
            self.build_listeners()
            self.register_user_event_listener(self.on_user_event)
        except Exception as e:
            print(f"--- Frontend initialisation error: {e} ---")
            self.send("exit program", 1)
            return

        # 2. Kickstart the loop
        self.scene.canvas.request_draw(self.one_frame)

        # 3. Blocking event loop: handles events and calls the custom one_frame().
        loop.run()

        # 4. notify backend of exit
        self.send("exit program", 1)

    def start(self):
        process = multiprocessing.Process(
            target = self.routine,
            args   = (),
            name   = "Frontend Process",
            daemon = False,
        )
        process.start()
        return process
    
    def on_backend_ping(self, data):
        print(f"Frontend received backend ping with data: {data}")

    def on_scatterplot_data(self, data):
        scatterplot = self.pages['main'].get("My Scatterplot")
        scatterplot.receive_data(data)