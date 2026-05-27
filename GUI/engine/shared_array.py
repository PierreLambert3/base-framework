# Wrapper around a Communications-managed shared-memory numpy block.
#
# IMPORTANT: this is the ONLY interface to a shared block. The raw ndarray
# is never returned by any public method (it would pin the underlying
# memoryview and make growth fail with BufferError).
#
# Reads:  copy_to_arr(out, src_index=())
# Writes: copy_to_buffer(src, dst_index=None) -- auto-reallocates when src
#         does not fit and dst_index is None (growth_factor=1.1 by default).
# Manual: reallocate(new_shape) (owner only).
# Escape hatch: with sa.write_view() as buf: ...  -- lends out the raw
#         ndarray for one operation (CUDA D->H, etc.). Reallocate is locked
#         out for the duration; the view is invalid after the with block.

import contextlib
import numpy as np


class SharedArray:
    """Owner if name is None (allocates); else attaches to an existing block."""

    def __init__(self, comms, key, shape, dtype, name=None, on_reallocated=None):
        self._comms          = comms
        self._key            = key
        self._owner          = name is None
        self._on_reallocated = on_reallocated
        self._lent           = 0
        if self._owner:
            self._ptr = comms.create_shared_array(key, shape, dtype)
        else:
            self._ptr = comms.attach_shared_array(key, name, shape, dtype)

    @property
    def shape(self):
        return self._ptr.shape

    @property
    def dtype(self):
        return self._ptr.dtype

    def copy_to_arr(self, out, src_index=()):
        """Copy self[src_index] into out (no view returned)."""
        out[...] = self._ptr[src_index]

    def copy_to_buffer(self, src, dst_index=None, growth_factor=1.1):
        """Copy src into the shared block. Owner only.
        If dst_index is None: write src at the leading-shape slice, auto-
        reallocating (per-axis) when src does not fit; new shape is
        ceil(src.shape * growth_factor) element-wise.
        If dst_index is given: direct write at that index, no grow."""
        assert self._owner, "copy_to_buffer() called on attached SharedArray"
        if dst_index is None:
            fits = (len(src.shape) == len(self._ptr.shape)
                    and all(s <= S for s, S in zip(src.shape, self._ptr.shape)))
            if not fits:
                new_shape = tuple(int(np.ceil(s * growth_factor)) for s in src.shape)
                self.reallocate(new_shape)
            self._ptr[tuple(slice(0, s) for s in src.shape)] = src
        else:
            self._ptr[dst_index] = src

    def reallocate(self, new_shape):
        """Owner: release current block, allocate a new one under the same
        key, notify peers via on_reallocated (if set)."""
        assert self._owner, "reallocate() called on attached SharedArray"
        assert self._lent == 0, "reallocate() called while a write_view is lent"
        dtype = self._ptr.dtype
        self._ptr = None
        self._comms.release_shared_array(self._key)
        self._ptr = self._comms.create_shared_array(self._key, new_shape, dtype)
        if self._on_reallocated is not None:
            info = self._comms.get_shared_array_info(self._key)
            self._on_reallocated(info)

    def _reattach(self, name, new_shape):
        """Attached side: drop view, re-attach to a new block."""
        assert not self._owner, "_reattach() called on owner SharedArray"
        dtype = self._ptr.dtype
        self._ptr = None
        self._ptr = self._comms.attach_shared_array(self._key, name, new_shape, dtype)

    def close(self):
        self._ptr = None
        self._comms.release_shared_array(self._key)

    @contextlib.contextmanager
    def write_view(self):
        """Yield the raw ndarray for in-place writes (e.g. CUDA D->H out= buffer).
        The view MUST NOT be retained past the with block; reallocate is
        blocked while any view is lent."""
        self._lent += 1
        try:
            yield self._ptr
        finally:
            self._lent -= 1
