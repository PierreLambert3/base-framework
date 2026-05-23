from GUI.engine.frontend.graphical_elements.graphical_element import Element_2d, Container
from GUI.engine.frontend.graphical_elements.parallelepiped import Parallelepiped
from GUI.engine.frontend.theme import ORANGE_YELLOW, ORANGE_DARK, interpolate_color, brighten, darken, PINK_NEON, BONE, PINK_ELECTRIC, ORANGE_RED, DARK_HIGHLIGHT
from GUI.engine.frontend import theme as _theme
from GUI.engine.frontend.audio import play_hover_in, play_hover_out
import pygfx
import numpy as np


BLACK = "#000000"


class ButtonColourScheme:
    """Defines the border, text, and background colours for each visual state of a button.
    
    Each state is a (border_colour, text_colour, background_colour) tuple.
    A default scheme is derived automatically from a base ``colour`` / ``text_colour``
    / ``background_colour`` triple so that existing call-sites keep working without
    changes.

    When hovered the text and background colours are swapped by default.
    """

    def __init__(self, colour, text_colour, background_colour=BLACK, *,
                 base=None, active=None, hovered=None, clicking=None, unclickable=None):
        # -- base (idle) -------------------------------------------------------
        if base is not None:
            self.base_border, self.base_text, self.base_bg = base
        else:
            self.base_border = colour
            self.base_text   = text_colour
            self.base_bg     = background_colour

        # -- active (toggled on, not hovered) ----------------------------------
        if active is not None:
            self.active_border, self.active_text, self.active_bg = active
        else:
            # self.active_border = brighten(interpolate_color(colour, BONE, 0.2), 0.2)
            self.active_border = brighten(colour, 0.4)
            self.active_text   = brighten(colour, 0.35)
            self.active_bg     = background_colour

        # -- hovered (text and background swap by default) ---------------------
        if hovered is not None:
            self.hovered_border, self.hovered_text, self.hovered_bg = hovered
        else:
            self.hovered_border = brighten(interpolate_color(colour, BONE, 0.5), 0.05)
            self.hovered_text   = brighten(interpolate_color(colour, BONE, 0.5), 0.05)
            self.hovered_bg     = background_colour

        # -- clicking (mouse-down, before release) -----------------------------
        if clicking is not None:
            self.clicking_border, self.clicking_text, self.clicking_bg = clicking
        else:
            self.clicking_border = background_colour
            self.clicking_text   = background_colour
            self.clicking_bg     = brighten(interpolate_color(colour, BONE, 0.2), 0.2)

        # -- unclickable -------------------------------------------------------
        if unclickable is not None:
            self.unclickable_border, self.unclickable_text, self.unclickable_bg = unclickable
        else:
            self.unclickable_border = darken(colour, 0.6)
            self.unclickable_text   = darken(colour, 0.6)
            self.unclickable_bg     = darken(colour, 0.7)


class Button_2d(Element_2d):
    """A 2D button with borders and centered text.
    
    The button supports five visual states, each with an explicit colour pair
    (border, text) grouped in a :class:`ButtonColourScheme`:

    * **base** – default idle look.
    * **active** – toggled-on (only meaningful when ``toggleable=True``).
    * **hovered** – pointer is over the button.
    * **clicking** – between pointer-down and pointer-up.
    * **unclickable** – the button is disabled and ignores interactions.

    The button can be switched between *clickable* and *unclickable* at any time
    via :meth:`set_clickable`.  When unclickable the button stays registered in
    the page's hoverable / clickable lists but simply does not respond to events.
    """
    
    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel, text, 
                 colour=None, text_colour=ORANGE_YELLOW, font_size=24,
                 background_colour=BLACK,
                 pointer_move_inside_callback=None, pointer_click_callback=None,
                 toggleable=False, clickable=True, colour_scheme=None, bold=False):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.text = text
        self.text_colour = text_colour
        self.font_size = font_size
        self.background_colour = background_colour
        self.hovered = False
        self.toggleable = toggleable
        self.pressed    = False
        self.clickable  = clickable

        # Colour scheme – build from explicit overrides or derive from base colours
        if colour_scheme is not None:
            self.colours = colour_scheme
        else:
            self.colours = ButtonColourScheme(self.colour, self.text_colour, self.background_colour)
        
        # Create background rectangle (drawn first so it sits behind border & text)
        self._make_background()

        # Create borders (all 4 sides)
        self._make_button_borders()
        
        # Create centered text
        self._make_text(bold)

        # Apply initial visual state
        if not self.clickable:
            self._apply_style(self.colours.unclickable_border, self.colours.unclickable_text, self.colours.unclickable_bg)

        # buttons attract cursor particles by default
        self.enable_particle_magnet()

        # register as hoverable and clickable
        self.register_hoverable()
        self.register_clickable()

        # Callbacks
        if pointer_move_inside_callback is not None:
            self.add_pointer_move_inside_callback(pointer_move_inside_callback)
        if pointer_click_callback is not None:
            self.add_pointer_click_callback(pointer_click_callback)

    def set_clickable(self, clickable: bool):
        """Enable or disable the button.  When unclickable the button keeps its
        registrations but ignores hover / click events and shows the
        *unclickable* colour scheme."""
        self.clickable = clickable
        if not clickable:
            self.hovered = False
            self._apply_style(self.colours.unclickable_border, self.colours.unclickable_text, self.colours.unclickable_bg)
        else:
            # Restore to the correct resting state
            if self.toggleable and self.pressed:
                self._apply_style(self.colours.active_border, self.colours.active_text, self.colours.active_bg)
            else:
                self._apply_style(self.colours.base_border, self.colours.base_text, self.colours.base_bg)

    def _apply_style(self, border_colour, text_colour, bg_colour=None):
        self.set_lines_colour(border_colour)
        self.text_obj.material.color    = text_colour
        if bg_colour is not None:
            self.bg_mesh.material.color = bg_colour

    def _make_background(self):
        """Create a filled rectangle behind the button."""
        w, h = self.size[0], self.size[1]
        geom = pygfx.plane_geometry(width=w, height=h)
        mat  = pygfx.MeshBasicMaterial(color=self.background_colour)
        self.bg_mesh = pygfx.Mesh(geom, mat)
        # Push slightly behind border/text to avoid z-fighting
        self.bg_mesh.local.position = (self.center[0], self.center[1], self.center[2] - 0.01)
        self.register_gfx_object(self.bg_mesh)

    def _make_button_borders(self, thickness=2.0):
        hw, hh = self.size[0] / 2, self.size[1] / 2
        bl = (-hw, -hh, 0)
        br = ( hw, -hh, 0)
        tr = ( hw,  hh, 0)
        tl = (-hw,  hh, 0)
        
        segments = [(bl, br), (br, tr), (tr, tl), (tl, bl)]
        self.add_lines(segments, colour=self.colour, thickness=thickness)
    
    def _make_text(self, bold=False):
        self.text_obj = pygfx.Text(
            text=self.text,
            font_size=self.font_size,
            anchor="middle-center",
            screen_space=False,
            material=pygfx.TextMaterial(color=self.text_colour)
        )
        if bold:
            self.text_obj.set_markdown(f"**{self.text}**")
        self.text_obj.local.position = self.center
        self.register_gfx_object(self.text_obj)
    
    def set_text(self, new_text):
        self.text = new_text
        self.text_obj.set_text(new_text)
    
    def die(self):
        self.scene.remove(self.bg_mesh)
        self.scene.remove(self.text_obj)
        super().die()

    # -- event handlers --------------------------------------------------------

    def on_pointer_move_inside(self, event, page_coords):
        if not self.clickable:
            return
        if not self.hovered:  # Only play sound on first hover-in
            play_hover_in()
        self.hovered = True
        if self.toggleable and self.pressed:
            # Swap text ↔ background (same logic as non-active hover)
            active_hovered_border = self.colours.active_bg
            active_hovered_text   = self.colours.active_bg
            active_hovered_bg     = self.colours.active_text
            self._apply_style(active_hovered_border, active_hovered_text, active_hovered_bg)
        else:
            self._apply_style(self.colours.hovered_border, self.colours.hovered_text, self.colours.hovered_bg)
        super().on_pointer_move_inside(event, page_coords)

    def on_pointer_move_outside(self, event, page_coords):
        if not self.clickable:
            return
        if self.hovered:  # Only play sound on first hover-out
            play_hover_out()
        self.hovered = False
        if self.toggleable and self.pressed:
            self._apply_style(self.colours.active_border, self.colours.active_text, self.colours.active_bg)
        else:
            self._apply_style(self.colours.base_border, self.colours.base_text, self.colours.base_bg)
        super().on_pointer_move_outside(event, page_coords)

    def on_pointer_down_inside(self, event, page_coords):
        if not self.clickable:
            return
        self._apply_style(self.colours.clicking_border, self.colours.clicking_text, self.colours.clicking_bg)
        super().on_pointer_down_inside(event, page_coords)

    def on_pointer_up_inside(self, event, page_coords):
        if not self.clickable:
            return
        if self.toggleable:
            self.toggle()
        super().on_pointer_up_inside(event, page_coords)
    
    def toggle(self):
        self.set_pressed(not self.pressed)
    
    def set_pressed(self, pressed_state):
        self.pressed = pressed_state
        if not self.clickable:
            self._apply_style(self.colours.unclickable_border, self.colours.unclickable_text, self.colours.unclickable_bg)
        elif self.pressed:
            self._apply_style(self.colours.active_border, self.colours.active_text, self.colours.active_bg)
        else:
            self._apply_style(self.colours.base_border, self.colours.base_text, self.colours.base_bg)

    def stop_pointer_down_effect(self):
        if not self.clickable:
            return
        if self.toggleable and self.pressed:
            self._apply_style(self.colours.active_border, self.colours.active_text, self.colours.active_bg)
            return

        if self.hovered:
            self._apply_style(self.colours.hovered_border, self.colours.hovered_text, self.colours.hovered_bg)
        else:
            self._apply_style(self.colours.base_border, self.colours.base_text, self.colours.base_bg)

    def reset_colours(self):
        """Reset to the appropriate resting state (respects clickable & pressed)."""
        if not self.clickable:
            self._apply_style(self.colours.unclickable_border, self.colours.unclickable_text, self.colours.unclickable_bg)
        elif self.toggleable and self.pressed:
            self._apply_style(self.colours.active_border, self.colours.active_text, self.colours.active_bg)
        else:
            self._apply_style(self.colours.base_border, self.colours.base_text, self.colours.base_bg)


class Toggle_2d(Container):
    """A binary toggle made of two side-by-side buttons.

    Any click on either button switches the active selection to the other side.
    """

    def __init__(self, unique_name, parent, bl_xy_rel, size_xy_rel,
                 text_left, text_right,
                 colour=None, text_colour=ORANGE_YELLOW, font_size=24,
                 background_colour=BLACK,
                 initial_value=0, pointer_toggle_callback=None,
                 colour_scheme=None, bold=False, clickable=True):
        super().__init__(unique_name, parent, bl_xy_rel, size_xy_rel, colour=colour)
        self.value = initial_value
        self._texts = (text_left, text_right)
        self._toggle_callback = pointer_toggle_callback
        self._text_colour = text_colour if text_colour is not None else ORANGE_YELLOW

        shared = dict(colour=colour, text_colour=text_colour, font_size=font_size,
                      background_colour=background_colour, toggleable=True,
                      colour_scheme=colour_scheme, bold=bold, clickable=clickable)

        margin = 0.01  # horizontal gap between the two buttons (relative to toggle width)
        half = (1.0 - margin) / 2.0
        self.button_left = Button_2d(
            unique_name + "_left", self, (0.0, 0.0), (half, 1.0), text_left,
            pointer_click_callback=lambda e, b, p: self._on_click(e, p), **shared)
        self.button_right = Button_2d(
            unique_name + "_right", self, (half + margin, 0.0), (half, 1.0), text_right,
            pointer_click_callback=lambda e, b, p: self._on_click(e, p), **shared)

        self._apply_toggle_visuals()

    def _on_click(self, event, page_coords):
        new_value = 1 - self.value
        self._set_value_internal(new_value)
        if self._toggle_callback is not None:
            self._toggle_callback(event, self, new_value)

    def _set_value_internal(self, value):
        self.value = value
        self._apply_toggle_visuals()

    def _apply_toggle_visuals(self):
        active_btn   = self.button_left if self.value == 0 else self.button_right
        inactive_btn = self.button_right if self.value == 0 else self.button_left
        active_btn.set_pressed(True)
        inactive_btn.set_pressed(False)
        inactive_btn.text_obj.material.color = darken(self._text_colour, 0.5)

    def get_value(self, as_text=False):
        if as_text:
            return self._texts[self.value]
        return self.value

    def set_value(self, value):
        self._set_value_internal(value)

    def set_clickable(self, clickable: bool):
        self.button_left.set_clickable(clickable)
        self.button_right.set_clickable(clickable)