import pygfx 
from rendercanvas.auto import RenderCanvas, loop

class Scene:
    def __init__(self, canvas_title):
        self.canvas   = RenderCanvas(title=canvas_title)
        self.renderer = pygfx.WgpuRenderer(self.canvas)
        self.scene    = pygfx.Scene()

        # Camera (simple perspective)
        self.camera = pygfx.PerspectiveCamera(60, int(16 / 9))
        self.camera.local.position = (0, 0, 4)
        self.scene.add(self.camera)

    def setup_one_frame(self, one_frame_method):
        self.canvas.request_draw(one_frame_method)

    def render(self):
        self.renderer.render(self.scene, self.camera)