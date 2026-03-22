"""
Apple Neural Engine (ANE) training backend for flexynesis.

Provides ANELinear — a drop-in replacement for nn.Linear that offloads
matrix multiplications to the ANE via reverse-engineered private APIs.

Usage:
    from flexynesis.ane import is_available, swap_linear_layers, count_linear_depth

    if is_available():
        model = swap_linear_layers(model)
"""

def is_available() -> bool:
    """Return True if ANE bridge can be loaded and initialised."""
    try:
        from .bridge import init, compile_kernel, free_kernel
        from . import mil
        init()
        try:
            mt = mil.dynamic_linear(64, 64, 64)
            ib = mil.input_bytes(64, 64, 64)
            ob = mil.output_bytes(64, 64)
            handle = compile_kernel(mt, ib, ob)
            free_kernel(handle)
            return True
        except RuntimeError:
            return False
    except Exception:
        return False


from .linear import ANELinear, swap_linear_layers


def count_linear_depth(module) -> int:
    """Count ANELinear layers in a module (used for loss-scale calculation)."""
    from .linear import ANELinear
    return sum(1 for m in module.modules() if isinstance(m, ANELinear))
