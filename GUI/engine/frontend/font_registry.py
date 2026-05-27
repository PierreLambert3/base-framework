import os

_FONTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), 'fonts'))

# Cache: font_name -> resolved pygfx family string.
_loaded: dict[str, str] = {}


def get_family(font_name: str | None) -> str | None:
    """
    return the pygfx font-family name for font_name, loading it on first call.
    returns None if .ttf not found or font_name is None.

    Call example:

    family = get_family(self.font_name)
    text_kwargs = dict(
        text="text to show",
        font_size=26,
        anchor="middle-center",
        screen_space=False,
        material=pygfx.TextMaterial(color=colour),
    )
    if family is not None:
        text_kwargs['family'] = family
    self.text_obj = pygfx.Text(**text_kwargs)
    self.text_obj.local.position = self.center
    self.register_gfx_object(self.text_obj)

    """
    if font_name is None:
        return None
    if font_name in _loaded:
        return _loaded[font_name]
    font_path = os.path.join(_FONTS_DIR, font_name + '.ttf')
    if not os.path.isfile(font_path):
        print(f"Warning: font '{font_name}.ttf' not found in fonts folder, using default font.")
        return None
    from pygfx.utils.text import font_manager
    ff = font_manager.add_font_file(font_path)
    _loaded[font_name] = ff.family
    return ff.family
