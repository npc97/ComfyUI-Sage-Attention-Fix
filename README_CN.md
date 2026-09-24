
# ComfyUI Sage Attention Fix

## 简介

如果你开启了全局Sage Attention，但是出于各种原因并不希望在model的加载链中使用 `Patch Sage Attention KJ` 、 `MiniMax H3 Mem Eff Sage Attention Patch` 以及其他任何注意力节点，那么你很有可能会遇到 [SageAttention issue #386](https://github.com/thu-ml/SageAttention/issues/386) 这个问题，从而导致生成的视频里出现花屏。

这个节点只负责规避这个ISSUE，而不做任何额外的处理。

## 详细说明

`Sage Attention Fix` 是一个作用范围严格受限的 ComfyUI 模型补丁，用于规避 [SageAttention issue #386](https://github.com/thu-ml/SageAttention/issues/386)。

本节点不会启用 SageAttention，也不会选择或改变 SageAttention 模式，它会保留ComfyUI 当前选择的 Attention 后端和所有调用参数。

当实际后端是 ComfyUI 通过 `--use-sage-attention` 启用的全局 Sage Attention 时，本节点会在进入该后端前检查Q/K/V 的最大可达元素偏移。只有当非连续视图的地址范围达到 `2**31`、存在 int32 地址计算溢出风险时，才会将对应张量复制为 contiguous 存储。

如果张量变为 contiguous 后仍然超出有符号 int32 的安全范围，本节点会明确报错，而不是继续运行并承担静默损坏输出的风险。

## 安装

`git clone https://github.com/npc97/ComfyUI-Sage-Attention-Fix`

将文件夹复制到 `ComfyUI/custom_nodes`，然后重启 ComfyUI。

## 放置位置

应将本节点放在同样会拦截 Attention 的模型补丁之后、采样器之前。例如，和 `SelfLiftH3TST` 一起使用时，应采用以下顺序，以保留两个补丁：

```text
SelfLiftH3TST -> Sage Attention Fix -> MiniMax H3 Low VRAM Attention -> MiniMax H3 ChunkFeedForward
```

如果 `model` 和 `model_hires` 两条分支都会参与采样，则两条分支都需要应用本节点。

不要将本节点与 `Patch Sage Attention KJ` 或 `MiniMax H3 Mem Eff Sage Attention Patch` 同时使用。这些节点会主动替换 Attention 实现，而本节点的设计目标是完整保留 ComfyUI 的全局 Sage Attention 实现。

## 与其他 Attention override 的兼容性

ComfyUI 为每个模型提供一个 `optimized_attention_override` 插槽。本节点可以与遵循“中间件式”调用约定的 override 组合：override 接收上游提供的 Attention 函数，并在完成自己的检查或 Q/K/V 修改后调用该函数，例如：

```python
def override(func, q, k, v, *args, **kwargs):
    # 检查或修改 Q/K/V。
    return func(q, k, v, *args, **kwargs)
```

Fix 会包装传给现有 override 的函数，因此现有 override 会先完成其 Q/K/V 修改；随后，溢出检查会在控制权回到 ComfyUI 原始全局 Attention 后端之前立即执行。

这并不代表本节点能够保证与所有自定义 Attention override 兼容：

- 正常调用、按条件调用或多次调用传入 `func` 的 override 可以兼容；每次进入原始后端前都会执行安全检查。
- 某条执行路径没有调用 `func` 时，该路径不会进入原始后端，因此 Fix 也不会在该路径上运行。
- 忽略 `func`、直接调用自有 Attention 或 SageAttention 实现的 override 会完全绕过本节点的保护。
- 依赖通过自定义 `container_function` 接收 `AttentionTensorContainer` 的特殊 override 无法保证兼容，因为本节点会使 ComfyUI 在分发调用链前先取出普通张量。
- 在本节点之后应用的其他节点可能直接覆盖唯一的 override 插槽，而不是与其组合。因此，`Sage Attention Fix` 应放在 `SelfLiftH3TST` 等可协作 Attention 补丁之后。

简而言之，本节点可以与遵循 ComfyUI 协作式 `func` 调用约定的 override 组合，但无法对任意替换或绕过该约定的 override 承诺兼容性。
