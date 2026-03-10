"""Tests for SpeculativeSampler.

These tests use simple mock models to verify the rejection sampling
properties without requiring real pretrained weights.
"""

import numpy as np
from keras import ops

from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler
from keras_hub.src.tests.test_case import TestCase

# ---------------------------------------------------------------------------
# Mock CausalLM helpers
# ---------------------------------------------------------------------------


class _MockCache:
    """Lightweight container mimicking what _build_cache returns."""

    def __init__(self, batch_size, num_layers, seq_len, vocab_size):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        # flat tensor used as "cache" for the mock
        self._data = ops.zeros([batch_size, num_layers, 2, seq_len, 1, 1])


def _make_mock_model(vocab_size, always_predict=None):
    """Return a minimal object with the same interface as a CausalLM.

    Args:
        vocab_size: int.
        always_predict: int or None. If set, this token id is always
            returned with near-certainty (greedy). Otherwise, uniform.
    """

    class MockModel:
        def __init__(self):
            self.vocab_size = vocab_size
            self._always = always_predict

        def _build_cache(self, token_ids):
            batch_size = ops.shape(token_ids)[0]
            cache = ops.zeros([batch_size, 1, 2, ops.shape(token_ids)[1], 1, 1])
            return None, cache, [None]

        def call_with_cache(
            self, token_ids, cache, cache_update_index, eviction_masks=None
        ):
            batch_size = ops.shape(token_ids)[0]
            if self._always is not None:
                # One-hot logits strongly favouring `_always`
                logits = ops.zeros([batch_size, 1, self.vocab_size])
                update = ops.full([batch_size, 1, 1], 1e6)
                logits = ops.slice_update(logits, [0, 0, self._always], update)
            else:
                logits = ops.ones([batch_size, 1, self.vocab_size])
            hidden = ops.zeros([batch_size, 1, 16])
            return logits, hidden, cache, eviction_masks or [None]

    return MockModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class SpeculativeSamplerTest(TestCase):
    def setUp(self):
        super().setUp()
        self.vocab_size = 20
        self.batch_size = 2
        self.seq_len = 12

    # -------------------------------------------------------------------
    # Constructor
    # -------------------------------------------------------------------

    def test_init_defaults(self):
        draft = _make_mock_model(self.vocab_size)
        sampler = SpeculativeSampler(draft_model=draft)
        self.assertEqual(sampler.num_speculative_tokens, 4)
        self.assertEqual(sampler.temperature, 1.0)

    def test_init_custom(self):
        draft = _make_mock_model(self.vocab_size)
        sampler = SpeculativeSampler(
            draft_model=draft,
            num_speculative_tokens=3,
            temperature=0.8,
            seed=42,
        )
        self.assertEqual(sampler.num_speculative_tokens, 3)
        self.assertAlmostEqual(sampler.temperature, 0.8)

    # -------------------------------------------------------------------
    # Draft proposal
    # -------------------------------------------------------------------

    def test_draft_propose_shape(self):
        """_draft_propose returns correct shapes."""
        draft = _make_mock_model(self.vocab_size, always_predict=7)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=3
        )

        token_ids = ops.zeros([self.batch_size, self.seq_len], dtype="int32")
        _, draft_cache, draft_eviction = draft._build_cache(token_ids)
        draft_cache_info = (None, draft_cache, draft_eviction)

        draft_tokens, draft_lp, _ = sampler._draft_propose(
            draft_cache_info, token_ids, start_index=4, K=3
        )
        self.assertEqual(
            draft_tokens.shape,
            (self.batch_size, 3),
            f"Expected [{self.batch_size}, 3], got {draft_tokens.shape}",
        )
        self.assertEqual(draft_lp.shape, (self.batch_size, 3, self.vocab_size))

    def test_draft_always_predicts_correct_token(self):
        """When draft model always predicts token 7, proposals should be 7."""
        draft = _make_mock_model(self.vocab_size, always_predict=7)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=4
        )

        token_ids = ops.zeros([self.batch_size, self.seq_len], dtype="int32")
        _, draft_cache, draft_eviction = draft._build_cache(token_ids)
        draft_cache_info = (None, draft_cache, draft_eviction)

        draft_tokens, _, _ = sampler._draft_propose(
            draft_cache_info, token_ids, start_index=4, K=4
        )
        draft_np = ops.convert_to_numpy(draft_tokens)
        self.assertTrue(
            (draft_np == 7).all(),
            f"Expected all draft proposals == 7, got {draft_np}",
        )

    # -------------------------------------------------------------------
    # Target verify
    # -------------------------------------------------------------------

    def test_target_verify_log_prob_shape(self):
        """_target_verify returns [B, K+1, V] log-probs."""
        draft = _make_mock_model(self.vocab_size)
        target = _make_mock_model(self.vocab_size, always_predict=3)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=4
        )

        token_ids = ops.zeros([self.batch_size, self.seq_len], dtype="int32")
        draft_tokens = ops.zeros([self.batch_size, 4], dtype="int32")
        _, target_cache, target_eviction = target._build_cache(token_ids)

        lp, _, _ = sampler._target_verify(
            target,
            token_ids,
            draft_tokens,
            start_index=4,
            cache=target_cache,
            eviction_masks=target_eviction,
        )
        self.assertEqual(lp.shape, (self.batch_size, 5, self.vocab_size))

    # -------------------------------------------------------------------
    # Rejection sampling properties
    # -------------------------------------------------------------------

    def test_rejection_sample_output_shape(self):
        """_rejection_sample returns [B, K+1] accepted_tokens."""
        draft = _make_mock_model(self.vocab_size)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=4, seed=0
        )
        K = 4
        draft_tokens = ops.zeros([self.batch_size, K], dtype="int32")
        uniform_lp = ops.full(
            [self.batch_size, K, self.vocab_size],
            -np.log(self.vocab_size),
            dtype="float32",
        )
        target_uniform_lp = ops.full(
            [self.batch_size, K + 1, self.vocab_size],
            -np.log(self.vocab_size),
            dtype="float32",
        )
        accepted, n_accepted = sampler._rejection_sample(
            draft_tokens, uniform_lp, target_uniform_lp
        )
        self.assertEqual(accepted.shape, (self.batch_size, K + 1))
        n_acc_np = ops.convert_to_numpy(n_accepted)
        self.assertTrue((n_acc_np >= 0).all())
        self.assertTrue((n_acc_np <= K).all())

    def test_perfect_draft_accepts_all(self):
        """When draft matches target exactly, all tokens are accepted
        (n_accepted == K)."""
        draft = _make_mock_model(self.vocab_size, always_predict=5)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=4, seed=0
        )
        K = 4
        # Both draft and target confidently predict token 5
        draft_lp = ops.full(
            [self.batch_size, K, self.vocab_size], -1e9, dtype="float32"
        )
        # inject high prob at token 5
        draft_lp = ops.slice_update(
            draft_lp,
            [0, 0, 5],
            ops.full([self.batch_size, K, 1], 0.0),
        )
        target_lp = ops.full(
            [self.batch_size, K + 1, self.vocab_size], -1e9, dtype="float32"
        )
        target_lp = ops.slice_update(
            target_lp,
            [0, 0, 5],
            ops.full([self.batch_size, K + 1, 1], 0.0),
        )
        draft_tokens = ops.full([self.batch_size, K], 5, dtype="int32")

        _, n_accepted = sampler._rejection_sample(
            draft_tokens, draft_lp, target_lp
        )
        n_acc_np = ops.convert_to_numpy(n_accepted)
        self.assertTrue(
            (n_acc_np == K).all(),
            f"Expected all K={K} tokens accepted, got {n_acc_np}",
        )

    def test_completely_wrong_draft_accepts_none(self):
        """When draft prob >> target prob at draft token, few are accepted."""
        draft = _make_mock_model(self.vocab_size)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=4, seed=0
        )
        K = 4
        # Draft is very confident about token 0; target is uniform
        draft_lp = ops.full(
            [self.batch_size, K, self.vocab_size], -1e9, dtype="float32"
        )
        draft_lp = ops.slice_update(
            draft_lp,
            [0, 0, 0],
            ops.full([self.batch_size, K, 1], 0.0),
        )
        # Target uniform
        target_lp = ops.full(
            [self.batch_size, K + 1, self.vocab_size],
            -np.log(self.vocab_size),
            dtype="float32",
        )
        draft_tokens = ops.zeros([self.batch_size, K], dtype="int32")

        _, n_accepted = sampler._rejection_sample(
            draft_tokens, draft_lp, target_lp
        )
        n_acc_np = ops.convert_to_numpy(n_accepted)
        # Acceptance probability ≈ 1/V — very low, so most runs accept 0 tokens
        self.assertTrue(
            (n_acc_np <= 1).all(),
            f"Expected ≤1 accepted, got {n_acc_np}",
        )

    # -------------------------------------------------------------------
    # get_config
    # -------------------------------------------------------------------

    def test_get_config(self):
        draft = _make_mock_model(self.vocab_size)
        sampler = SpeculativeSampler(
            draft_model=draft, num_speculative_tokens=6, temperature=0.9, seed=7
        )
        cfg = sampler.get_config()
        self.assertEqual(cfg["num_speculative_tokens"], 6)
        self.assertAlmostEqual(cfg["temperature"], 0.9)
        self.assertEqual(cfg["seed"], 7)
