"""
ANELinear — drop-in replacement for nn.Linear using Apple Neural Engine.

Forward pass:  y = x @ W          (on ANE, fp16)
Backward dx:   dx = grad @ W.T    (on ANE, fp16)
Backward dW:   dW = x.T @ grad    (on CPU, fp32)
Bias:          y += b             (on CPU, fp32)

Key design decisions from the ANE training blog:
  1. Dynamic weights: compiled once per (IC, OC, batch) shape at first use.
     Subsequent steps just update IOSurface data — zero recompilation.
  2. Separate backward kernel: dx uses a kernel with IC/OC swapped and W^T staged.
  3. dW stays on CPU: weight gradients are x.T @ grad, done cheaply with numpy.
  4. Loss scaling: caller is responsible for scaling loss by (256 * depth)
     before .backward() to avoid fp16 gradient underflow in deep networks.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional

from . import bridge, mil


# Global kernel cache: (ic, oc, batch) → opaque ANE handle (int)
# Shared across all ANELinear instances so identical layer shapes reuse kernels.
_kernel_cache: dict[tuple[int, int, int], int] = {}

# Track total ANE compile count for diagnostics
_compile_count = 0


_ALIGN = 64  # ANE requires tensor dimensions to be multiples of 64


def _pad(n: int) -> int:
    """Round n up to the nearest multiple of _ALIGN."""
    return ((n + _ALIGN - 1) // _ALIGN) * _ALIGN


def _get_kernel(ic: int, oc: int, batch: int) -> int:
    """Return a compiled ANE kernel for this (IC, OC, batch) shape, compiling if needed."""
    global _compile_count
    key = (ic, oc, batch)
    if key not in _kernel_cache:
        bridge.init()
        mil_text = mil.dynamic_linear(ic, oc, batch)
        in_bytes  = mil.input_bytes(ic, oc, batch)
        out_bytes = mil.output_bytes(oc, batch)
        handle = bridge.compile_kernel(mil_text, in_bytes, out_bytes)
        _kernel_cache[key] = handle
        _compile_count += 1
    return _kernel_cache[key]


def _run_matmul(ic: int, oc: int, batch: int,
                x_fp32: np.ndarray, w_fp32: np.ndarray) -> np.ndarray:
    """
    Execute  output = x @ w  on ANE.

    x_fp32 : [batch, ic]  float32
    w_fp32 : [ic,    oc]  float32   (kernel-friendly layout)
    returns: [batch, oc]  float32

    Pads IC, OC, and batch to multiples of 64 as required by the ANE,
    then slices back to the true output shape.
    """
    assert x_fp32.shape == (batch, ic),  f"x shape mismatch: {x_fp32.shape} vs ({batch},{ic})"
    assert w_fp32.shape == (ic,    oc),  f"w shape mismatch: {w_fp32.shape} vs ({ic},{oc})"

    ic_p    = _pad(ic)
    oc_p    = _pad(oc)
    batch_p = _pad(batch)

    # Pad activations to [batch_p, ic_p] and weights to [ic_p, oc_p]
    x_p = np.zeros((batch_p, ic_p), dtype=np.float32)
    w_p = np.zeros((ic_p,    oc_p), dtype=np.float32)
    x_p[:batch, :ic] = x_fp32
    w_p[:ic,    :oc] = w_fp32

    # Pack into [IC_p, batch_p + OC_p] fp16
    sp = batch_p + oc_p
    packed = np.empty((ic_p, sp), dtype=np.float16)
    packed[:, :batch_p] = x_p.T.astype(np.float16)
    packed[:, batch_p:] = w_p.astype(np.float16)

    kernel  = _get_kernel(ic_p, oc_p, batch_p)
    raw_out = bridge.run_kernel(kernel, packed.flatten().tobytes(),
                                mil.output_bytes(oc_p, batch_p))

    # Output is [OC_p, batch_p] fp16 → slice to [oc, batch] → transpose to [batch, oc]
    out_fp16 = np.frombuffer(raw_out, dtype=np.float16).reshape(oc_p, batch_p)
    return out_fp16[:oc, :batch].T.astype(np.float32)


# ── Autograd Function ────────────────────────────────────────────────────────

class _ANELinearFn(torch.autograd.Function):
    """
    Custom autograd function for ANE-accelerated linear layer.

    Forward:    y  = x @ W           (ANE)
    Backward:   dx = grad @ W.T      (ANE, backward kernel with IC/OC swapped)
                dW = x.T @ grad      (CPU numpy)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x:      [batch, ic]
        # weight: [ic, oc]  (stored transposed relative to nn.Linear convention)
        batch, ic = x.shape
        oc = weight.shape[1]

        x_np = x.contiguous().numpy()
        w_np = weight.contiguous().numpy()

        y_np = _run_matmul(ic, oc, batch, x_np, w_np)

        ctx.save_for_backward(x.detach(), weight.detach())
        ctx.shapes = (ic, oc, batch)

        return torch.from_numpy(y_np)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, weight = ctx.saved_tensors
        ic, oc, batch = ctx.shapes

        grad_np = grad_output.contiguous().numpy()  # [batch, oc]
        w_np    = weight.contiguous().numpy()        # [ic, oc]

        # dx = grad @ W.T  →  use backward kernel (IC=oc, OC=ic) with W^T staged
        # W.T has shape [oc, ic] → passed as w for the (oc→ic) kernel
        w_t_np = w_np.T.copy()                       # [oc, ic]
        dx_np  = _run_matmul(oc, ic, batch, grad_np, w_t_np)   # [batch, ic]

        # dW = x.T @ grad  on CPU  → [ic, batch] @ [batch, oc] = [ic, oc]
        dW_np  = (x.contiguous().numpy().T @ grad_np).astype(np.float32)

        return torch.from_numpy(dx_np), torch.from_numpy(dW_np)


# ── Module ───────────────────────────────────────────────────────────────────

class ANELinear(nn.Module):
    """
    Drop-in replacement for nn.Linear using Apple Neural Engine for matmul.

    Weight is stored as [in_features, out_features] (kernel-friendly layout),
    which is the transpose of nn.Linear's [out_features, in_features].

    Usage:
        layer = ANELinear(512, 64)
        # or convert from existing nn.Linear:
        layer = ANELinear.from_linear(existing_linear)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # [in_features, out_features] — transposed from nn.Linear convention
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias   = nn.Parameter(torch.zeros(out_features)) if bias else None

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier uniform, same as nn.Linear's default
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure CPU (ANE bridge requires CPU tensors)
        if x.device.type != "cpu":
            x = x.cpu()

        out = _ANELinearFn.apply(x, self.weight)

        if self.bias is not None:
            out = out + self.bias

        return out

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "ANELinear":
        """
        Convert a trained nn.Linear to ANELinear.
        Handles the weight transpose (nn.Linear stores [out, in], we store [in, out]).
        """
        m = cls(linear.in_features, linear.out_features, linear.bias is not None)
        with torch.no_grad():
            m.weight.copy_(linear.weight.T)  # [out,in].T → [in,out]
            if m.bias is not None and linear.bias is not None:
                m.bias.copy_(linear.bias)
        return m

    def to_linear(self) -> nn.Linear:
        """Convert back to nn.Linear (for inference with standard tools)."""
        lin = nn.Linear(self.in_features, self.out_features, self.bias is not None)
        with torch.no_grad():
            lin.weight.copy_(self.weight.T)
            if lin.bias is not None and self.bias is not None:
                lin.bias.copy_(self.bias)
        return lin

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, bias={self.bias is not None}, backend=ANE"


def swap_linear_layers(module: nn.Module, min_features: int = 128) -> nn.Module:
    """
    Recursively replace all nn.Linear layers with ANELinear in-place.

    min_features: skip layers smaller than this (ANE has minimum size requirements).
    Returns the modified module.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if child.in_features >= min_features and child.out_features >= min_features:
                setattr(module, name, ANELinear.from_linear(child))
            # else: leave small layers on CPU as nn.Linear
        else:
            swap_linear_layers(child, min_features)
    return module


def ane_compile_stats() -> dict:
    """Return diagnostics about compiled kernels."""
    return {
        "kernels_compiled": _compile_count,
        "unique_shapes": len(_kernel_cache),
        "shapes": list(_kernel_cache.keys()),
    }
