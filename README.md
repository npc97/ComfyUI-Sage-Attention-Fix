
# ComfyUI Sage Attention Fix

[中文说明](README_CN.md)

## Introduction

If you have enabled global Sage Attention but, for any reason, do not want to use `Patch Sage Attention KJ`, `MiniMax H3 Mem Eff Sage Attention Patch` and any other attentions in the model loading chain, you may encounter [SageAttention issue #386](https://github.com/thu-ml/SageAttention/issues/386), which can cause corrupted frames in generated videos.

This node only guards against that issue and performs no additional processing.

## Details

`Sage Attention Fix` is a narrowly scoped ComfyUI model patch for [SageAttention issue #386](https://github.com/thu-ml/SageAttention/issues/386).

It does not enable SageAttention or select a SageAttention mode. It preserves the attention backend and all arguments selected by ComfyUI. When the active backend is ComfyUI's global `--use-sage-attention` backend, the node checks the reachable element offset of Q/K/V immediately before the backend call. Only a view whose strided address range reaches `2**31` is copied to contiguous storage.

If a contiguous tensor would still exceed the signed-int32-safe range, the node raises an error rather than allowing possible silent output corruption.

## Installation

`git clone https://github.com/npc97/ComfyUI-Sage-Attention-Fix`

Copy or this directory into `ComfyUI/custom_nodes`, then restart ComfyUI.

## Placement

Place the node after model patches that also intercept attention, and before the sampler. For example, when using it with `SelfLiftH3TST`, use the following order so both patches are preserved:

```text
SelfLiftH3TST -> Sage Attention Fix -> MiniMax H3 Low VRAM Attention -> MiniMax H3 ChunkFeedForward
```

Apply it to both `model` and `model_hires` branches when both branches sample.

Do not combine it with `Patch Sage Attention KJ` or `MiniMax H3 Mem Eff Sage Attention Patch`; those nodes intentionally replace the attention implementation, while this node is designed to retain the global ComfyUI SageAttention implementation unchanged.

## Compatibility with other attention overrides

ComfyUI exposes one `optimized_attention_override` slot per model. This node can compose with middleware-style overrides that accept the supplied attention function and eventually call it:

```python
def override(func, q, k, v, *args, **kwargs):
    # Inspect or modify Q/K/V.
    return func(q, k, v, *args, **kwargs)
```

The fix wraps the function passed to the existing override, so any Q/K/V edits made by that override occur first. The overflow check then runs immediately before control returns to ComfyUI's original global attention backend.

This is not a guarantee of compatibility with every custom attention override:

- An override that calls the supplied `func` normally, conditionally, or more than once is compatible; every call to the original backend is guarded.
- An override that returns without calling `func` does not reach the original backend on that path, so the guard does not run there.
- An override that ignores `func` and directly invokes its own attention or SageAttention implementation bypasses this guard entirely.
- An override that depends on receiving `AttentionTensorContainer` through a custom `container_function` is not guaranteed to be compatible, because this node causes ComfyUI to unwrap the containers before dispatching the chain.
- A node applied after this one may replace the single override slot instead of composing with it. Place `Sage Attention Fix` after cooperative attention patches such as `SelfLiftH3TST`.

In short, this node composes with overrides that follow ComfyUI's cooperative `func`-calling convention. It cannot safely promise compatibility with an arbitrary override that replaces or bypasses that convention.
