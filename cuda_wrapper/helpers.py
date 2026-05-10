from typing import List, Tuple, Any
import numpy as np
from pathlib import Path
import re
import ast
import operator as _op

__MAX_UINT32__ = np.uint32(np.iinfo(np.uint32).max)

rng = np.random.default_rng()

def generate_uint32_seed():
    return rng.integers(0, __MAX_UINT32__, dtype=np.uint32)

def next_seed_uint32(seed):
    x = np.uint32(seed)
    x ^= np.uint32(x << np.uint32(13))
    x ^= np.uint32(x >> np.uint32(17))
    x ^= np.uint32(x << np.uint32(5))
    return np.uint32(x)


_C_DEFINE_RE = re.compile(r"^\s*#define\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\b(?P<rest>.*)$")


def _safe_eval_simple_expr(expr: str, names: dict):
    """Safely evaluate a small numeric/string expression.

    Supports literals, names, and basic operators. Rejects calls, attribute access,
    subscripts, comprehensions, etc.
    """

    allowed_binops = {
        ast.Add: _op.add,
        ast.Sub: _op.sub,
        ast.Mult: _op.mul,
        ast.Div: _op.truediv,
        ast.FloorDiv: _op.floordiv,
        ast.Mod: _op.mod,
        ast.Pow: _op.pow,
        ast.BitOr: _op.or_,
        ast.BitAnd: _op.and_,
        ast.BitXor: _op.xor,
        ast.LShift: _op.lshift,
        ast.RShift: _op.rshift,
    }
    allowed_unaryops = {
        ast.UAdd: _op.pos,
        ast.USub: _op.neg,
        ast.Invert: _op.invert,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise KeyError(node.id)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in allowed_binops:
                raise ValueError(f"Operator not allowed: {op_type.__name__}")
            return allowed_binops[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in allowed_unaryops:
                raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
            return allowed_unaryops[op_type](_eval(node.operand))
        if isinstance(node, ast.Expr):
            return _eval(node.value)
        # Anything else (Call, Attribute, Subscript, Compare, etc.) is disallowed.
        raise ValueError(f"Expression node not allowed: {type(node).__name__}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree.body)


def read_cuda_defines(file_path):
    """Read `#define` constants from a CUDA/C-like file.

    Only supports object-like macros:
      - `#define NAME VALUE`
      - `#define NAME` (stored as True)

    Function-like macros (e.g. `#define FOO(x) ...`) are ignored.

    Returns: dict[str, Any]
    """
    path = Path(file_path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    defines = {}
    for raw_line in text.splitlines():
        m = _C_DEFINE_RE.match(raw_line)
        if not m:
            continue

        name = m.group("name")
        rest = (m.group("rest") or "").lstrip()

        # Skip function-like macros: NAME(...)
        if rest.startswith("("):
            continue

        # Strip inline comments.
        value_str = rest
        value_str = value_str.split("//", 1)[0]
        value_str = re.sub(r"/\*.*?\*/", "", value_str)
        value_str = value_str.strip()

        if value_str == "":
            defines[name] = True
            continue

        # Normalize common C/CUDA suffixes and booleans.
        value_norm = value_str
        value_norm = re.sub(r"(?<=\d)[fF]\b", "", value_norm)  # 10.0f -> 10.0
        value_norm = re.sub(r"(?<=\d)[uUlL]+\b", "", value_norm)  # 10u, 10UL -> 10
        value_norm = re.sub(r"\btrue\b", "True", value_norm)
        value_norm = re.sub(r"\bfalse\b", "False", value_norm)

        # Try to interpret as a Python literal first (fast path).
        parsed = None
        try:
            parsed = ast.literal_eval(value_norm)
        except Exception:
            pass

        # Fall back to a safe expression evaluator to allow simple arithmetic or
        # references to previously parsed defines.
        if parsed is None:
            try:
                parsed = _safe_eval_simple_expr(value_norm, defines)
            except Exception:
                parsed = value_str  # keep original if we can't parse it safely

        defines[name] = parsed

    return defines


class DoubleBufferPair:
    """
    Represents a pair of GPU arrays for double buffering.
    
    Attributes:
        name: Base name for the buffers (creates {name}_now and {name}_prev).
        A, B: The two GPU arrays.
        copy_on_swap: If True, copy prev→now after each swap (for persistent state).
    """
    __slots__ = ('name', 'A', 'B', 'copy_on_swap')
    
    def __init__(self, name: str, buffer_A, buffer_B, copy_on_swap: bool = False):
        self.name = name
        self.A = buffer_A
        self.B = buffer_B
        self.copy_on_swap = copy_on_swap


class DoubleBuffer:
    """
    This code was generated by a LLM.
    Manages double-buffered GPU arrays with synchronized phase toggling.
    
    Two modes are supported:
    
    1. SWAP-ONLY (for spikes, etc.):
       - Just swap pointers: what was 'now' becomes 'prev'
       - Use: register(name, values)
    
    2. COPY-ON-SWAP (for charges, potentials, etc.):
       - After swapping, copy prev→now so 'now' starts with the previous state
       - Useful for sparse updates or accumulative patterns
       - Use: register_with_copy(name, values)
    
    Usage:
        self._double_buffer = DoubleBuffer(cuda_ctx, owner=self)
        
        # Swap-only (spikes): just need previous step's values for reading
        self._double_buffer.register("spikes", np.zeros((n,), dtype=np.uint8))
        
        # Copy-on-swap (charges): 'now' must start with 'prev' values
        self._double_buffer.register_with_copy("charges", np.zeros((n,), dtype=np.float32))
        
        # At start of each step (pass stream for async copy):
        self._double_buffer.swap(stream)
    """
    
    def __init__(self, cuda_ctx, owner):
        self.cuda_ctx = cuda_ctx
        self.owner = owner
        self.phase_A = True
        self._pairs: List[DoubleBufferPair] = []
    
    def register(self, name: str, values: np.ndarray) -> Tuple[Any, Any]:
        return self._register_pair(name, values, copy_on_swap=False)
    
    def register_with_copy(self, name: str, values: np.ndarray) -> Tuple[Any, Any]:
        return self._register_pair(name, values, copy_on_swap=True)
    
    def _register_pair(self, name: str, values: np.ndarray, copy_on_swap: bool) -> Tuple[Any, Any]:
        buffer_A = self.cuda_ctx.m(values.copy())
        buffer_B = self.cuda_ctx.m(values.copy())
        pair = DoubleBufferPair(name, buffer_A, buffer_B, copy_on_swap)
        self._pairs.append(pair)
        self._set_pair_attributes(pair)
        return getattr(self.owner, f"{name}_now"), getattr(self.owner, f"{name}_prev")
    
    def _set_pair_attributes(self, pair: DoubleBufferPair):
        """Set _now and _prev attributes based on current phase."""
        if self.phase_A:
            setattr(self.owner, f"{pair.name}_now", pair.A)
            setattr(self.owner, f"{pair.name}_prev", pair.B)
        else:
            setattr(self.owner, f"{pair.name}_now", pair.B)
            setattr(self.owner, f"{pair.name}_prev", pair.A)
    
    def swap(self, stream=None):
        self.phase_A = not self.phase_A
        for pair in self._pairs:
            self._set_pair_attributes(pair)
        
            if pair.copy_on_swap:
                now  = getattr(self.owner, f"{pair.name}_now")
                prev = getattr(self.owner, f"{pair.name}_prev")
                self.cuda_ctx.copy_d2d(src=prev, dst=now, stream=stream)