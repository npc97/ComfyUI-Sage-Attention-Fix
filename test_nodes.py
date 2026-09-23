import unittest

import torch

from nodes import (
    _INT32_ADDRESS_LIMIT,
    _build_guard,
    _max_reachable_element_offset,
)


class OffsetCalculationTests(unittest.TestCase):
    def test_small_contiguous_tensor_is_safe(self):
        self.assertEqual(
            _max_reachable_element_offset((1, 56, 1024, 128), (7340032, 131072, 128, 1)),
            56 * 1024 * 128 - 1,
        )

    def test_h3_fused_qkv_view_crosses_issue_386_limit(self):
        sequence = 99866
        heads = 56
        head_dim = 128
        fused_sequence_stride = 3 * heads * head_dim
        offset = _max_reachable_element_offset(
            (1, heads, sequence, head_dim),
            (sequence * fused_sequence_stride, head_dim, fused_sequence_stride, 1),
        )
        self.assertGreaterEqual(offset, _INT32_ADDRESS_LIMIT)

    def test_same_h3_shape_is_safe_when_contiguous(self):
        sequence = 99866
        heads = 56
        head_dim = 128
        offset = _max_reachable_element_offset(
            (1, heads, sequence, head_dim),
            (heads * sequence * head_dim, sequence * head_dim, head_dim, 1),
        )
        self.assertLess(offset, _INT32_ADDRESS_LIMIT)


class OverrideCompositionTests(unittest.TestCase):
    @staticmethod
    def _global_sage_stub(q, k, v, *args, **kwargs):
        return q, k, v, args, kwargs

    def setUp(self):
        self._global_sage_stub.__name__ = "attention_sage"
        self._global_sage_stub.__module__ = "comfy.ldm.modules.attention"

    def test_preserves_the_original_backend_and_arguments(self):
        q = torch.zeros(1, 2, 3, 4)
        k = torch.ones_like(q)
        v = torch.full_like(q, 2)
        guard = _build_guard(None)

        result = guard(
            self._global_sage_stub,
            q,
            k,
            v,
            2,
            mask="mask-value",
            smooth_k=False,
        )

        self.assertIs(result[0], q)
        self.assertIs(result[1], k)
        self.assertIs(result[2], v)
        self.assertEqual(result[3], (2,))
        self.assertEqual(result[4]["mask"], "mask-value")
        self.assertIs(result[4]["smooth_k"], False)

    def test_wraps_an_existing_override_instead_of_replacing_it(self):
        calls = []

        def previous_override(func, q, k, v, *args, **kwargs):
            calls.append("previous")
            return func(q + 1, k, v, *args, **kwargs)

        q = torch.zeros(1, 1, 1, 1)
        guard = _build_guard(previous_override)
        result = guard(self._global_sage_stub, q, q, q, 1)

        self.assertEqual(calls, ["previous"])
        self.assertEqual(float(result[0].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
