"""A narrowly scoped guard for SageAttention issue #386.

The node preserves ComfyUI's selected attention implementation.  It only makes
Q/K/V contiguous when their strided address range can overflow a signed int32
index in affected SageAttention kernels.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import torch


_INT32_ADDRESS_LIMIT = 1 << 31
_FIX_MARKER = "_sage_attention_issue_386_fix"


def _max_reachable_element_offset(
    shape: Sequence[int], stride: Sequence[int]
) -> int:
    """Return the largest relative element offset reachable by a strided view."""

    if len(shape) != len(stride):
        raise ValueError("shape and stride must have the same number of dimensions")

    return sum(
        (int(size) - 1) * abs(int(step))
        for size, step in zip(shape, stride)
        if int(size) > 0
    )


def _has_int32_address_risk(tensor: Any) -> bool:
    """Whether SageAttention's int32 index arithmetic can overflow for tensor."""

    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and tensor.numel() > 0
        and _max_reachable_element_offset(tensor.shape, tensor.stride())
        >= _INT32_ADDRESS_LIMIT
    )


def _is_comfy_global_sage_attention(attention: Callable[..., Any]) -> bool:
    """Match ComfyUI's global --use-sage-attention backend, not other patches."""

    return (
        getattr(attention, "__name__", "") == "attention_sage"
        and getattr(attention, "__module__", "")
        == "comfy.ldm.modules.attention"
    )


def _make_safe(tensor: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if not _has_int32_address_risk(tensor):
        return tensor, False

    safe_tensor = tensor.contiguous()
    if _has_int32_address_risk(safe_tensor):
        raise RuntimeError(
            "Sage Attention Fix: this attention tensor remains outside the "
            "signed-int32-safe address range even after making it contiguous. "
            "Reduce the sequence length or split the attention into smaller "
            "head groups; continuing could silently corrupt the output."
        )
    return safe_tensor, True


def _build_guard(
    previous_override: Callable[..., Any] | None,
) -> Callable[..., Any]:
    state = {"copy_logged": False, "inactive_logged": False}

    def guard_override(
        original_attention: Callable[..., Any],
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        def guarded_original(
            inner_q: torch.Tensor,
            inner_k: torch.Tensor,
            inner_v: torch.Tensor,
            *inner_args: Any,
            **inner_kwargs: Any,
        ) -> Any:
            # Preserve every argument and the exact backend selected by ComfyUI.
            # For non-Sage backends this node is a true pass-through.
            if not _is_comfy_global_sage_attention(original_attention):
                if not state["inactive_logged"]:
                    logging.info(
                        "Sage Attention Fix is inactive because the current "
                        "attention backend is %s.%s, not ComfyUI's global Sage "
                        "Attention backend.",
                        getattr(original_attention, "__module__", "<unknown>"),
                        getattr(original_attention, "__name__", "<unknown>"),
                    )
                    state["inactive_logged"] = True
                return original_attention(
                    inner_q, inner_k, inner_v, *inner_args, **inner_kwargs
                )

            fixed_q, copied_q = _make_safe(inner_q)
            fixed_k, copied_k = _make_safe(inner_k)
            fixed_v, copied_v = _make_safe(inner_v)

            if (copied_q or copied_k or copied_v) and not state["copy_logged"]:
                logging.info(
                    "Sage Attention Fix applied the issue #386 contiguous "
                    "guard to an unsafe Q/K/V view."
                )
                state["copy_logged"] = True

            return original_attention(
                fixed_q, fixed_k, fixed_v, *inner_args, **inner_kwargs
            )

        # SelfLiftH3TST and similar model patches also use this override slot.
        # Keep them outside the guard so their Q/K edits happen first, then
        # sanitize the final views immediately before the original backend.
        if previous_override is not None:
            return previous_override(
                guarded_original, q, k, v, *args, **kwargs
            )
        return guarded_original(q, k, v, *args, **kwargs)

    setattr(guard_override, _FIX_MARKER, True)
    return guard_override


class SageAttentionFix:
    """Protect ComfyUI global SageAttention from issue #386 only."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"
    DESCRIPTION = (
        "Preserves ComfyUI's global Sage Attention backend and settings. "
        "Only makes unsafe strided Q/K/V views contiguous when their address "
        "range can trigger SageAttention issue #386 (signed int32 overflow)."
    )

    def patch(self, model):
        patched = model.clone()
        transformer_options = patched.model_options.setdefault(
            "transformer_options", {}
        )
        previous_override = transformer_options.get(
            "optimized_attention_override"
        )

        # Applying the node more than once should not add nested copies or logs.
        if getattr(previous_override, _FIX_MARKER, False):
            return (patched,)

        transformer_options["optimized_attention_override"] = _build_guard(
            previous_override
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"SageAttentionFix": SageAttentionFix}
NODE_DISPLAY_NAME_MAPPINGS = {"SageAttentionFix": "Sage Attention Fix"}
