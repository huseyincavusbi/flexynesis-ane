"""
ctypes bindings for libane_bridge.dylib
"""

import ctypes
import os
import numpy as np
from pathlib import Path

# Resolve dylib: check package-local copy first, then ANE repo sibling directory
_DYLIB = Path(__file__).parent / "libane_bridge.dylib"
if not _DYLIB.exists():
    _DYLIB = Path(__file__).parent.parent.parent.parent / "ANE" / "bridge" / "libane_bridge.dylib"

_lib = None
_initialized = False


def _load():
    global _lib
    if _lib is not None:
        return _lib

    if not _DYLIB.exists():
        raise FileNotFoundError(
            f"libane_bridge.dylib not found at {_DYLIB}\n"
            "Run: cd ANE/bridge && make"
        )

    _lib = ctypes.CDLL(str(_DYLIB))

    _lib.ane_bridge_init.restype = ctypes.c_int
    _lib.ane_bridge_init.argtypes = []

    _lib.ane_bridge_compile.restype = ctypes.c_void_p
    _lib.ane_bridge_compile.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_void_p, ctypes.c_size_t,
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int, ctypes.POINTER(ctypes.c_size_t),
    ]

    _lib.ane_bridge_eval.restype = ctypes.c_bool
    _lib.ane_bridge_eval.argtypes = [ctypes.c_void_p]

    _lib.ane_bridge_write_input.restype = None
    _lib.ane_bridge_write_input.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]

    _lib.ane_bridge_read_output.restype = None
    _lib.ane_bridge_read_output.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t
    ]

    _lib.ane_bridge_free.restype = None
    _lib.ane_bridge_free.argtypes = [ctypes.c_void_p]

    _lib.ane_bridge_get_compile_count.restype = ctypes.c_int
    _lib.ane_bridge_get_compile_count.argtypes = []

    return _lib


def init():
    global _initialized
    lib = _load()
    if not _initialized:
        ret = lib.ane_bridge_init()
        if ret != 0:
            raise RuntimeError("ane_bridge_init() failed — AppleNeuralEngine.framework not accessible")
        _initialized = True


def compile_kernel(mil_text: str, input_bytes: int, output_bytes: int) -> int:
    """
    Compile a MIL program to an ANE kernel.
    Returns an opaque handle (int) to the kernel.
    """
    lib = _load()
    mil_bytes = mil_text.encode("utf-8")
    in_sizes  = (ctypes.c_size_t * 1)(input_bytes)
    out_sizes = (ctypes.c_size_t * 1)(output_bytes)

    handle = lib.ane_bridge_compile(
        mil_bytes, len(mil_bytes),
        None, 0,
        1, in_sizes,
        1, out_sizes,
    )
    if not handle:
        raise RuntimeError("ane_bridge_compile() returned NULL — check MIL syntax")
    return handle


def run_kernel(handle: int, input_data: bytes, output_size: int) -> bytes:
    """
    Write input, eval, read output. Returns raw output bytes.
    """
    lib = _load()
    lib.ane_bridge_write_input(handle, 0, input_data, len(input_data))

    ok = lib.ane_bridge_eval(handle)
    if not ok:
        raise RuntimeError("ane_bridge_eval() failed — ANE inference error")

    out = (ctypes.c_uint8 * output_size)()
    lib.ane_bridge_read_output(handle, 0, out, output_size)
    return bytes(out)


def free_kernel(handle: int):
    _load().ane_bridge_free(handle)


def compile_count() -> int:
    return _load().ane_bridge_get_compile_count()
