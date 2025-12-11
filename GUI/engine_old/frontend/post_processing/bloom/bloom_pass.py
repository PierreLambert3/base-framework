"""Bloom post-processing pass for pygfx (multi-pass dual-kawase style).

Pipeline order per frame:
    1. Threshold extract bright areas -> level0
    2. Downsample chain (level i -> i+1) with Kawase taps
    3. Upsample chain (level n -> 0) simple upscale (placeholder for tap-mix)
    4. Composite original scene + bloom into provided target

Parameters (runtime adjustable via set_params):
    threshold (float)     : brightness threshold
    knee (float)          : soft knee for smooth thresholding
    strength (float)      : bloom intensity blend factor
    color_select (bool)   : enable color selective weighting
    color_center (rgb)    : target color for selection
    color_sigma (float)   : gaussian sigma for color distance

Limitations / Future work:
    - Upsample currently overwrites (no additive energy normalization)
    - Could add multi-scale weights per level
    - Could optionally use mipmaps instead of manual pyramid
"""
from __future__ import annotations
import time
from typing import List, Tuple
import pygfx as gfx
from pygfx.renderers.wgpu.engine.effectpasses import EffectPass, FullQuadPass
import numpy as np

# ---------- Internal Passes ---------------------------------------------------

class _ThresholdPass(FullQuadPass):
    uniform_type = dict(
        threshold="f4", knee="f4", use_color="f4",
        color_center="3xf4", color_sigma="f4",
    )
    wgsl = """
    @fragment
    fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
        let c = textureSample(colorTex, texSampler, varyings.texCoord);
        let luma = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));
        let x = max(luma - u_effect.threshold, 0.0);
        // Soft knee shaping
        let bright = x * x / (u_effect.knee + 1e-6 + x * x);
        
        var out_color = c.rgb * bright;
        if (u_effect.use_color > 0.5) {
            // Tint mode: blend the bloom towards the selected color
            // The brighter the pixel, the more it gets tinted with color_center
            let tint_strength = u_effect.color_sigma;  // repurpose sigma as tint strength (0-1)
            let tinted = mix(c.rgb, u_effect.color_center * luma, tint_strength);
            out_color = tinted * bright;
        }
        return vec4<f32>(out_color, 1.0);
    }
    """

class _DownPass(FullQuadPass):
    uniform_type = dict(texel_parent="2xf4")
    wgsl = """
    const OFFSETS = array<vec2<i32>,8>(
        vec2<i32>( 1, 1), vec2<i32>(-1, 1), vec2<i32>( 1,-1), vec2<i32>(-1,-1),
        vec2<i32>( 2, 0), vec2<i32>(-2, 0), vec2<i32>( 0, 2), vec2<i32>( 0,-2)
    );
    @fragment
    fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
        var acc = vec3<f32>(0.0);
        for (var i: u32 = 0u; i < 8u; i = i + 1u) {
            let ofs = vec2<f32>(OFFSETS[i]) * u_effect.texel_parent;
            acc += textureSample(colorTex, texSampler, varyings.texCoord + ofs).rgb;
        }
        return vec4<f32>(acc / 8.0, 1.0);
    }
    """

class _UpPass(FullQuadPass):
    uniform_type = dict(texel_current="2xf4")
    wgsl = """
    @fragment
    fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
        // Simple fetch; could mix taps for smoother spread.
        let base = textureSample(colorTex, texSampler, varyings.texCoord).rgb;
        return vec4<f32>(base, 1.0);
    }
    """

class _CompositePass(FullQuadPass):
    uniform_type = dict(strength="f4")
    wgsl = """
    @fragment
    fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
        let scene = textureSample(colorTex, texSampler, varyings.texCoord).rgb;
        let glow  = textureSample(extraTex, texSampler, varyings.texCoord).rgb;
        let outc = scene + glow * u_effect.strength;
        return vec4<f32>(outc, 1.0);
    }
    """

# ---------- Public BloomPass --------------------------------------------------

class BloomPass(EffectPass):
    uniform_type = dict(
        EffectPass.uniform_type,
        threshold="f4", knee="f4", strength="f4", use_color="f4",
        color_center="3xf4", color_sigma="f4",
    )

    def __init__(
        self,
        depth: int = 5,
        threshold: float = 1.0,
        knee: float = 0.5,
        strength: float = 0.8,
        color_select: bool = False,
        color_center: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        color_sigma: float = 0.35,
        preferred_format: str = "rgba16float",
    ):
        super().__init__()
        import wgpu  # local import to avoid hard dependency at import time
        self._wgpu = wgpu
        self.depth = max(2, depth)
        self._preferred_format = preferred_format
        self._format_actual: str | None = None
        # Store underlying GPU textures + their views (lists kept parallel)
        # Internal GPU resources (lists kept parallel). We keep them as untyped lists
        # because wgpu types may not have stubs available in the environment.
        self._pyramid_textures = []  # list of GPUTexture
        self._pyramid_views = []     # list of GPUTextureView
        self._size_cache = (-1, -1)
        # Root uniforms (EffectPass already created buffer)
        self._uniform_data["threshold"] = float(threshold)
        self._uniform_data["knee"] = float(knee)
        self._uniform_data["strength"] = float(strength)
        self._uniform_data["use_color"] = 1.0 if color_select else 0.0
        self._uniform_data["color_center"] = color_center
        self._uniform_data["color_sigma"] = float(color_sigma)
        # Internal passes (each a FullQuadPass subclass)
        self._threshold_pass = _ThresholdPass()
        self._down_passes = [_DownPass() for _ in range(self.depth - 1)]
        self._up_passes = [_UpPass() for _ in range(self.depth - 1)]
        self._composite_pass = _CompositePass()
        self.hdr_enabled = False
        self._dirty_params = True

    # Dummy fragment (unused) - required by EffectPass interface
    wgsl = "@fragment fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> { return vec4<f32>(0.0); }"

    def set_params(self, **kwargs):
        fields = self._uniform_data.dtype.fields or {}
        changed = False
        for k, v in kwargs.items():
            if k == "color_select":
                val = 1.0 if v else 0.0
                if self._uniform_data["use_color"] != val:
                    self._uniform_data["use_color"] = val
                    changed = True
            elif k in fields:
                current = self._uniform_data[k]
                # Handle array comparison (e.g., color_center)
                is_array = isinstance(current, np.ndarray) and current.ndim > 0 and current.size > 1
                if is_array:
                    if not np.array_equal(current, v):
                        self._uniform_data[k] = v
                        changed = True
                elif current != v:
                    self._uniform_data[k] = v
                    changed = True
        if changed:
            self._dirty_params = True

    def _choose_format(self):
        if self._format_actual is not None:
            return
        candidates = []
        if self._preferred_format:
            candidates.append(self._preferred_format)
        candidates += ["rgba16float", "rgba8unorm"]
        uniq = []
        seen = set()
        for c in candidates:
            if c not in seen:
                uniq.append(c); seen.add(c)
        for fmt in uniq:
            try:
                # Probe via pygfx Texture (lightweight) to validate format
                gfx.Texture(None, dim=2, size=(1, 1), format=fmt)
                self._format_actual = fmt
                self.hdr_enabled = fmt != "rgba8unorm"
                return
            except Exception:
                continue
        self._format_actual = "rgba8unorm"
        self.hdr_enabled = False

    def _destroy_pyramid(self):
        # We simply drop references; renderer/device will GC when appropriate.
        self._pyramid_textures.clear()
        self._pyramid_views.clear()

    def _allocate_pyramid(self, w: int, h: int):
        self._destroy_pyramid()
        usage = self._wgpu.TextureUsage.RENDER_ATTACHMENT | self._wgpu.TextureUsage.TEXTURE_BINDING
        for i in range(self.depth):
            tw = max(1, w >> i); th = max(1, h >> i)
            tex = self._device.create_texture(  # type: ignore[arg-type]
                size=(tw, th, 1),
                usage=usage,  # type: ignore[arg-type]
                format=self._format_actual or "rgba8unorm",  # type: ignore[arg-type]
                mip_level_count=1,
                sample_count=1,
            )  # type: ignore[call-arg]
            view = tex.create_view()  # type: ignore[attr-defined]
            self._pyramid_textures.append(tex)
            self._pyramid_views.append(view)
        self._size_cache = (w, h)

    def _ensure_textures(self, w: int, h: int):
        if (w, h) == self._size_cache and self._pyramid_views:
            return
        self._choose_format()
        self._allocate_pyramid(w, h)

    def _sync_internal_uniforms(self):
        # Always sync uniforms to internal passes (in case of direct _uniform_data modification)
        self._threshold_pass._uniform_data["threshold"] = self._uniform_data["threshold"]
        self._threshold_pass._uniform_data["knee"] = self._uniform_data["knee"]
        self._threshold_pass._uniform_data["use_color"] = self._uniform_data["use_color"]
        self._threshold_pass._uniform_data["color_center"] = self._uniform_data["color_center"]
        self._threshold_pass._uniform_data["color_sigma"] = self._uniform_data["color_sigma"]
        self._composite_pass._uniform_data["strength"] = self._uniform_data["strength"]
        self._dirty_params = False

    def render(self, command_encoder, color_tex, depth_tex, target_tex):  # type: ignore[override]
        # time uniform for animation/noise compatibility
        self._uniform_data["time"] = time.perf_counter()
        # color_tex is a GPUTextureView from renderer
        w = color_tex.texture.width
        h = color_tex.texture.height
        if w <= 0 or h <= 0:
            super().render(command_encoder, color_tex, depth_tex, target_tex)
            return
        self._ensure_textures(w, h)
        self._sync_internal_uniforms()
        views = self._pyramid_views
        # 1. Threshold extract
        self._threshold_pass.render(command_encoder, colorTex=color_tex, targetTex=views[0])
        # 2. Downsample chain
        for i in range(self.depth - 1):
            src = views[i]; dst = views[i + 1]
            tw = src.texture.width; th = src.texture.height
            self._down_passes[i]._uniform_data["texel_parent"] = (1.0 / tw, 1.0 / th)
            self._down_passes[i].render(command_encoder, colorTex=src, targetTex=dst)
        # 3. Upsample chain
        for i in range(self.depth - 1, 0, -1):
            src = views[i]; dst = views[i - 1]
            tw = src.texture.width; th = src.texture.height
            self._up_passes[i - 1]._uniform_data["texel_current"] = (1.0 / tw, 1.0 / th)
            self._up_passes[i - 1].render(command_encoder, colorTex=src, targetTex=dst)
        # 4. Composite to final target
        self._composite_pass.render(
            command_encoder,
            colorTex=color_tex,
            extraTex=views[0],
            targetTex=target_tex,
        )

__all__ = ["BloomPass"]
