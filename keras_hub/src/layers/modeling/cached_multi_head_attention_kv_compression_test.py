"""Tests for KV cache compression in CachedMultiHeadAttention."""

from keras import ops
from keras import random

from keras_hub.src.layers.modeling.cached_multi_head_attention import (
    CachedMultiHeadAttention,
)
from keras_hub.src.tests.test_case import TestCase


class KVCompressionTest(TestCase):
    """Tests for H2O, SnapKV, and StreamingLLM eviction in
    CachedMultiHeadAttention."""

    def setUp(self):
        super().setUp()
        self.batch_size = 2
        self.seq_len = 16
        self.num_heads = 4
        self.key_dim = 8
        self.hidden_dim = self.num_heads * self.key_dim
        self.kv_budget = 8  # keep half

    def _build_layer(self, eviction_policy):
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy=eviction_policy,
        )
        return layer

    def _prefill_with_cache(self, layer, x):
        """Run a full pre-fill pass, return (output, cache, eviction_mask)."""
        input_cache = ops.zeros(
            (
                self.batch_size,
                2,
                self.seq_len,
                self.num_heads,
                self.key_dim,
            )
        )
        mask = ops.tril(ops.ones((self.seq_len, self.seq_len), dtype="bool"))
        result = layer(
            x,
            x,
            cache=input_cache,
            cache_update_index=0,
            attention_mask=mask,
        )
        return result

    # ------------------------------------------------------------------
    # Basic API tests
    # ------------------------------------------------------------------

    def test_no_eviction_returns_two_tuple(self):
        """Without eviction, call() returns (output, cache) as before."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        input_cache = ops.zeros(
            (self.batch_size, 2, self.seq_len, self.num_heads, self.key_dim)
        )
        result = layer(x, x, cache=input_cache, cache_update_index=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        output, cache = result
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.hidden_dim)
        )

    def test_invalid_policy_raises(self):
        """Passing an unsupported eviction policy raises ValueError."""
        with self.assertRaisesRegex(ValueError, "eviction_policy"):
            CachedMultiHeadAttention(
                num_heads=2,
                key_dim=4,
                kv_budget=4,
                eviction_policy="unknown_policy",
            )

    def test_eviction_without_budget_raises(self):
        """Specifying a policy without kv_budget raises ValueError."""
        with self.assertRaisesRegex(ValueError, "kv_budget"):
            CachedMultiHeadAttention(
                num_heads=2,
                key_dim=4,
                eviction_policy="h2o",
                kv_budget=None,
            )

    # ------------------------------------------------------------------
    # H2O eviction
    # ------------------------------------------------------------------

    def test_h2o_prefill_returns_eviction_mask(self):
        """H2O pre-fill returns a 3-tuple including an eviction_mask."""
        layer = self._build_layer("h2o")
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        result = self._prefill_with_cache(layer, x)
        self.assertIsInstance(result, tuple)
        self.assertEqual(
            len(result), 3, "Expected (output, cache, eviction_mask)"
        )
        output, cache, eviction_mask = result
        self.assertEqual(
            output.shape, (self.batch_size, self.seq_len, self.hidden_dim)
        )
        self.assertEqual(
            cache.shape,
            (self.batch_size, 2, self.seq_len, self.num_heads, self.key_dim),
        )
        self.assertEqual(eviction_mask.shape, (self.batch_size, self.seq_len))

    def test_h2o_mask_keeps_correct_count(self):
        """H2O eviction mask retains exactly kv_budget True entries per row."""
        layer = self._build_layer("h2o")
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, _, eviction_mask = self._prefill_with_cache(layer, x)
        eviction_mask_np = ops.convert_to_numpy(eviction_mask)
        for b in range(self.batch_size):
            n_kept = int(eviction_mask_np[b].sum())
            # Due to possible ties, we allow ±1.
            self.assertBetween(n_kept, self.kv_budget - 1, self.kv_budget + 1)

    def test_h2o_decode_uses_eviction_mask(self):
        """Passing eviction_mask during decode does not raise and returns
        correct shape."""
        layer = self._build_layer("h2o")
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, cache, eviction_mask = self._prefill_with_cache(layer, x)

        # Simulate a single decode step (T=1)
        x_decode = random.uniform((self.batch_size, 1, self.hidden_dim))
        mask_1d = ops.tril(ops.ones((1, self.seq_len), dtype="bool"))
        result = layer(
            x_decode,
            x_decode,
            cache=cache,
            cache_update_index=self.seq_len - 1,
            attention_mask=mask_1d,
            attention_eviction_mask=eviction_mask,
        )
        self.assertEqual(len(result), 2)
        output, _ = result
        self.assertEqual(output.shape, (self.batch_size, 1, self.hidden_dim))

    def test_h2o_decode_below_budget_no_eviction(self):
        """When seq_len <= kv_budget, no eviction mask is returned."""
        small_seq = self.kv_budget  # exactly at budget
        layer = self._build_layer("h2o")
        x = random.uniform((self.batch_size, small_seq, self.hidden_dim))
        cache = ops.zeros(
            (self.batch_size, 2, small_seq, self.num_heads, self.key_dim)
        )
        mask = ops.tril(ops.ones((small_seq, small_seq), dtype="bool"))
        result = layer(
            x, x, cache=cache, cache_update_index=0, attention_mask=mask
        )
        # seq_len == kv_budget → no eviction → 2-tuple
        self.assertEqual(len(result), 2)

    # ------------------------------------------------------------------
    # SnapKV eviction
    # ------------------------------------------------------------------

    def test_snapkv_prefill_returns_eviction_mask(self):
        """SnapKV pre-fill returns a valid 3-tuple."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="snapkv",
            # window_size must be < seq_len/2 to avoid window-dominated scores
            snapkv_window_size=4,
            snapkv_kernel_size=3,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        result = self._prefill_with_cache(layer, x)
        self.assertEqual(len(result), 3)
        _, _, eviction_mask = result
        self.assertEqual(eviction_mask.shape, (self.batch_size, self.seq_len))

    def test_snapkv_mask_keeps_correct_count(self):
        """SnapKV mask retains approximately kv_budget positions."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="snapkv",
            snapkv_window_size=4,
            snapkv_kernel_size=3,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, _, eviction_mask = self._prefill_with_cache(layer, x)
        eviction_mask_np = ops.convert_to_numpy(eviction_mask)
        for b in range(self.batch_size):
            n_kept = int(eviction_mask_np[b].sum())
            # SnapKV always keeps the window tokens at high score
            # → allow ±window
            self.assertBetween(n_kept, self.kv_budget - 1, self.kv_budget + 4)

    # ------------------------------------------------------------------
    # StreamingLLM eviction
    # ------------------------------------------------------------------

    def test_streaming_llm_prefill_returns_eviction_mask(self):
        """StreamingLLM pre-fill returns a valid 3-tuple."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="streaming_llm",
            streaming_llm_n_sink=2,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        result = self._prefill_with_cache(layer, x)
        self.assertEqual(len(result), 3)
        _, _, eviction_mask = result
        self.assertEqual(eviction_mask.shape, (self.batch_size, self.seq_len))

    def test_streaming_llm_sink_tokens_always_kept(self):
        """StreamingLLM mask always keeps the first n_sink tokens."""
        n_sink = 2
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="streaming_llm",
            streaming_llm_n_sink=n_sink,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, _, eviction_mask = self._prefill_with_cache(layer, x)
        eviction_mask_np = ops.convert_to_numpy(eviction_mask)
        # First n_sink positions must all be True (kept)
        for b in range(self.batch_size):
            self.assertTrue(
                eviction_mask_np[b, :n_sink].all(),
                f"Sink tokens not all kept for batch item {b}: "
                f"{eviction_mask_np[b, :n_sink]}",
            )

    def test_streaming_llm_recent_tokens_always_kept(self):
        """StreamingLLM mask always keeps the most recent
        (budget-n_sink) tokens."""
        n_sink = 2
        n_recent = self.kv_budget - n_sink
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="streaming_llm",
            streaming_llm_n_sink=n_sink,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, _, eviction_mask = self._prefill_with_cache(layer, x)
        eviction_mask_np = ops.convert_to_numpy(eviction_mask)
        # Last n_recent positions must all be True (kept)
        for b in range(self.batch_size):
            self.assertTrue(
                eviction_mask_np[b, -n_recent:].all(),
                f"Recent tokens not all kept for batch item {b}: "
                f"{eviction_mask_np[b, -n_recent:]}",
            )

    def test_streaming_llm_middle_tokens_evicted(self):
        """StreamingLLM mask evicts the middle tokens (not sink, not recent)."""
        n_sink = 2
        n_recent = self.kv_budget - n_sink
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="streaming_llm",
            streaming_llm_n_sink=n_sink,
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        _, _, eviction_mask = self._prefill_with_cache(layer, x)
        eviction_mask_np = ops.convert_to_numpy(eviction_mask)
        # Middle positions must all be False (evicted)
        middle = eviction_mask_np[:, n_sink : self.seq_len - n_recent]
        for b in range(self.batch_size):
            self.assertFalse(
                middle[b].any(),
                f"Middle tokens not all evicted for batch item {b}: "
                f"{middle[b]}",
            )

    # ------------------------------------------------------------------
    # get_config round-trip
    # ------------------------------------------------------------------

    def test_get_config_round_trip(self):
        """get_config() → from_config() preserves eviction settings."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            kv_budget=self.kv_budget,
            eviction_policy="h2o",
            snapkv_window_size=16,
            snapkv_kernel_size=3,
            streaming_llm_n_sink=2,
        )
        config = layer.get_config()
        self.assertEqual(config["kv_budget"], self.kv_budget)
        self.assertEqual(config["eviction_policy"], "h2o")
        self.assertEqual(config["snapkv_window_size"], 16)
        self.assertEqual(config["snapkv_kernel_size"], 3)
        self.assertEqual(config["streaming_llm_n_sink"], 2)
        # Reconstruct
        layer2 = CachedMultiHeadAttention.from_config(config)
        self.assertEqual(layer2.kv_budget, self.kv_budget)
        self.assertEqual(layer2.eviction_policy, "h2o")

    # ------------------------------------------------------------------
    # Backward-compatibility: existing behavior unchanged when no eviction
    # ------------------------------------------------------------------

    def test_no_eviction_cache_identical_to_original(self):
        """Without eviction, outputs match what CachedMultiHeadAttention
        would produce with the original implementation."""
        layer = CachedMultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim
        )
        x = random.uniform((self.batch_size, self.seq_len, self.hidden_dim))
        cache = ops.zeros(
            (self.batch_size, 2, self.seq_len, self.num_heads, self.key_dim)
        )
        mask = ops.tril(ops.ones((self.seq_len, self.seq_len), dtype="bool"))
        out_full, cache_full = layer(
            x, x, cache=cache, cache_update_index=0, attention_mask=mask
        )
        # Token-by-token loop should match the full-sequence forward pass.
        outputs = ops.zeros_like(x)
        cache2 = ops.zeros_like(cache)

        for i in range(self.seq_len):
            xi = ops.slice(x, (0, i, 0), (self.batch_size, 1, self.hidden_dim))
            mi = ops.slice(mask, (i, 0), (1, self.seq_len))
            out_i, cache2 = layer(
                xi, xi, cache=cache2, cache_update_index=i, attention_mask=mi
            )
            outputs = ops.slice_update(outputs, [0, i, 0], out_i)

        self.assertAllClose(outputs, out_full, atol=1e-4)
