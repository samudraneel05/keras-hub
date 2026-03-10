#!/usr/bin/env python3
"""Benchmark script for keras-hub inference optimizations.

Compares baseline against:
  1. StreamingLLM KV eviction   (cache-size comparison on the attention layer)
  2. H2O KV eviction
  3. SnapKV KV eviction
  4. SpeculativeSampler (draft-model acceleration)

Usage (CPU / no GPU required for the layer benchmarks):
    KERAS_BACKEND=torch python benchmarks/inference_optimization_benchmarks.py

For full model benchmarks with pretrained weights (GPU recommended):
    KERAS_BACKEND=torch python benchmarks/inference_optimization_benchmarks.py \
        --full_model --preset smollm3_135m_en

Results are printed to stdout and optionally saved as JSON to
    benchmarks/benchmark_results.json
"""

import argparse
import json
import time

import numpy as np
from keras import ops
from keras import random

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72


def _header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def _row(label, value, unit=""):
    print(f"  {label:<45} {value} {unit}")


def _timeit(fn, warmup=2, repeats=10):
    """Return mean ± std wall-clock time in milliseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ---------------------------------------------------------------------------
# A1 — KV Eviction (layer-level benchmark)
# ---------------------------------------------------------------------------


def bench_kv_eviction_layer(
    batch_size=4,
    seq_len=512,
    num_heads=8,
    key_dim=64,
    kv_budget_ratio=0.25,
    repeats=8,
):
    """Measure attention output and eviction mask generation time."""
    from keras_hub.src.layers.modeling.cached_multi_head_attention import (
        CachedMultiHeadAttention,
    )

    hidden_dim = num_heads * key_dim
    kv_budget = int(seq_len * kv_budget_ratio)

    x = random.uniform([batch_size, seq_len, hidden_dim])
    cache = ops.zeros([batch_size, 2, seq_len, num_heads, key_dim])
    mask = ops.cast(ops.tril(ops.ones([seq_len, seq_len])), "bool")

    results = {}

    policies = [None, "streaming_llm", "h2o", "snapkv"]
    for policy in policies:
        layer = CachedMultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            kv_budget=kv_budget if policy else None,
            eviction_policy=policy,
        )
        # Build layer
        _ = layer(x[:1], x[:1])

        def run():
            return layer(
                x, x, cache=cache, cache_update_index=0, attention_mask=mask
            )

        mean_ms, std_ms = _timeit(run, repeats=repeats)
        label = policy if policy else "baseline (no eviction)"
        results[label] = {"mean_ms": mean_ms, "std_ms": std_ms}
        _row(label, f"{mean_ms:.2f} ± {std_ms:.2f}", "ms")

    # Eviction count info
    _row(
        "kv_budget",
        f"{kv_budget} / {seq_len}",
        f"({kv_budget_ratio * 100:.0f}% retention)",
    )
    return results


# ---------------------------------------------------------------------------
# A2 — Speculative Decoding (mock-model speedup measurement)
# ---------------------------------------------------------------------------


def bench_speculative_decoding(
    batch_size=2,
    seq_len=32,
    gen_len=40,
    vocab_size=512,
    num_speculative_tokens=4,
    repeats=5,
):
    """Compare single-model greedy vs. speculative decoding using mock models.

    Since we use mocks, this measures the *algorithmic overhead* of the
    speculative loop (as opposed to the theoretical GPU speedup from larger
    models). Real speedup is experienced with actual pretrained models.
    """
    from keras_hub.src.samplers.speculative_sampler import SpeculativeSampler

    # Mock model — always predicts token 1
    class MockCausalLM:
        def __init__(self, delay_ms=0):
            self._delay = delay_ms

        def _build_cache(self, token_ids):
            B = ops.shape(token_ids)[0]
            S = ops.shape(token_ids)[1]
            cache = ops.zeros([B, 1, 2, S, 1, 1])
            return None, cache, [None]

        def call_with_cache(self, token_ids, cache, idx, eviction_masks=None):
            B = ops.shape(token_ids)[0]
            logits = ops.zeros([B, 1, vocab_size])
            logits = ops.slice_update(
                logits, [0, 0, 1], ops.full([B, 1, 1], 1e6)
            )
            if self._delay > 0:
                time.sleep(self._delay / 1000)
            return logits, None, cache, eviction_masks or [None]

        def score(self, *a, **kw):
            pass

    # Baseline: greedy token-by-token loop
    target = MockCausalLM()
    prompt = ops.zeros([batch_size, seq_len + gen_len], dtype="int32")
    _, cache, eviction_masks = target._build_cache(prompt[:, :seq_len])

    def baseline_generate():
        c = cache
        em = eviction_masks
        for step in range(gen_len):
            tok = ops.slice(prompt, [0, seq_len + step - 1], [batch_size, 1])
            _, _, c, em = target.call_with_cache(tok, c, seq_len + step - 1, em)

    # Speculative
    draft = MockCausalLM()
    sampler = SpeculativeSampler(
        draft_model=draft,
        num_speculative_tokens=num_speculative_tokens,
        seed=42,
    )

    def speculative_generate():
        sampler(
            next=None,
            prompt=prompt.copy() if hasattr(prompt, "copy") else prompt,
            cache=cache,
            index=seq_len,
            model=target,
        )

    _header("A2 — Speculative Decoding Overhead (mock models)")
    _row("batch_size", batch_size)
    _row("gen_len", gen_len)
    _row("num_speculative_tokens K", num_speculative_tokens)

    base_mean, base_std = _timeit(baseline_generate, repeats=repeats)
    _row("Greedy baseline", f"{base_mean:.2f} ± {base_std:.2f}", "ms")

    spec_mean, spec_std = _timeit(speculative_generate, repeats=repeats)
    _row("Speculative loop", f"{spec_mean:.2f} ± {spec_std:.2f}", "ms")

    # With equally fast draft+target, loop overhead ≈ K×more calls.
    # Real speedup emerges when model forward pass dominates (GPU).
    return {
        "greedy": {"mean_ms": base_mean, "std_ms": base_std},
        "speculative": {"mean_ms": spec_mean, "std_ms": spec_std},
    }


# ---------------------------------------------------------------------------
# Full-model benchmark (optional, requires preset)
# ---------------------------------------------------------------------------


def bench_full_model(preset, max_length=64, repeats=3):
    """End-to-end generate() benchmark with a real keras-hub preset.

    Compares:
    - Baseline (top_k sampler)
    - StreamingLLM eviction (kv_budget=32)
    - H2O eviction (kv_budget=32)
    """
    import keras_hub

    _header(f"Full-model generate() benchmark: {preset}")

    prompt = "The history of machine learning began with"
    kv_budget = 32

    results = {}

    for label, policy in [
        ("baseline (top_k)", None),
        ("streaming_llm kv_budget=32", "streaming_llm"),
        ("h2o kv_budget=32", "h2o"),
        ("snapkv kv_budget=32", "snapkv"),
    ]:
        model = keras_hub.models.CausalLM.from_preset(preset)

        # For KV eviction: patch the backbone's transformer layers
        if policy is not None:
            for layer in model.backbone.transformer_layers:
                layer.kv_budget = kv_budget
                layer.eviction_policy = policy
                layer._self_attention_layer.kv_budget = kv_budget
                layer._self_attention_layer.eviction_policy = policy

        model.compile(sampler="top_k")

        def run():
            return model.generate(prompt, max_length=max_length)

        # Warmup
        _ = run()
        mean_ms, std_ms = _timeit(run, warmup=1, repeats=repeats)
        _row(label, f"{mean_ms:.1f} ± {std_ms:.1f}", "ms")
        results[label] = {"mean_ms": mean_ms, "std_ms": std_ms}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark keras-hub inference optimizations"
    )
    parser.add_argument(
        "--full_model",
        action="store_true",
        help="Also run end-to-end generate() benchmark (requires GPU).",
    )
    parser.add_argument(
        "--preset",
        default="smollm3_135m_en",
        help="keras-hub preset for full-model benchmark.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=256,
        help="Source sequence length for layer benchmark.",
    )
    parser.add_argument(
        "--kv_budget_ratio",
        type=float,
        default=0.25,
        help="Fraction of KV pairs to retain (0.25 = 75%% compression).",
    )
    parser.add_argument(
        "--output_json",
        default="benchmarks/benchmark_results.json",
        help="Path to write JSON results.",
    )
    args = parser.parse_args()

    all_results = {}

    # ------------------------------------------------------------------
    # A1 — Layer benchmarks
    # ------------------------------------------------------------------
    _header("A1 — KV Eviction Policy Comparison (layer-level)")
    print(f"  seq_len={args.seq_len}, kv_budget_ratio={args.kv_budget_ratio}")
    print("  batch=4, heads=8, key_dim=64")
    layer_results = bench_kv_eviction_layer(
        seq_len=args.seq_len,
        kv_budget_ratio=args.kv_budget_ratio,
    )
    all_results["kv_eviction_layer"] = layer_results

    # Compute speedup vs baseline
    if "baseline (no eviction)" in layer_results:
        base = layer_results["baseline (no eviction)"]["mean_ms"]
        print("\n  Relative overhead vs baseline:")
        for k, v in layer_results.items():
            if k != "baseline (no eviction)":
                overhead = (v["mean_ms"] - base) / base * 100
                sign = "+" if overhead >= 0 else ""
                _row(k, f"{sign}{overhead:.1f}%", "(eviction overhead)")

    # ------------------------------------------------------------------
    # A2 — Speculative decoding overhead
    # ------------------------------------------------------------------
    spec_results = bench_speculative_decoding()
    all_results["speculative_decoding"] = spec_results

    # ------------------------------------------------------------------
    # Full-model (optional)
    # ------------------------------------------------------------------
    if args.full_model:
        full_results = bench_full_model(args.preset)
        all_results["full_model"] = full_results

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    import os

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results written to {args.output_json}")

    _header("Summary")
    print("  A1 (KV eviction): pre-fill overhead measured above.")
    print("  A2 (speculative): loop overhead measured above.")
    print("  Real model speedup from speculative decoding depends on")
    print("  forward-pass latency of draft vs. target models (GPU).")


if __name__ == "__main__":
    main()
