import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export


@keras_hub_export("keras_hub.layers.CachedMultiHeadAttention")
class CachedMultiHeadAttention(keras.layers.MultiHeadAttention):
    """MultiHeadAttention layer with cache support and KV compression.

    This layer is suitable for use in autoregressive decoding. It can be used
    to cache decoder self-attention and cross-attention. The forward pass
    can happen in one of three modes:

    - No cache, same as regular multi-head attention.
    - Static cache (`cache_update_index` is None). In this case, the
        cached key/value projections will be used and the input values will
        be ignored.
    - Updated cache (`cache_update_index` is not None). In this case, new
        key/value projections are computed using the input, and spliced into
        the cache at the specified index.

    Note that caching is useful only during inference and should not be used
    during training.

    KV Cache Compression (optional):
    When `kv_budget` and `eviction_policy` are set, the layer applies
    KV cache compression after the pre-fill pass. Three policies are supported:

    - `"h2o"`: Heavy-Hitter Oracle. Keeps the `kv_budget` tokens with the
      highest accumulated attention score over all queries.
    - `"snapkv"`: Keeps tokens scoring highest by the windowed attention of
      the last `snapkv_window_size` queries (smoothed with average pooling).
    - `"streaming_llm"`: Keeps the first `streaming_llm_n_sink` tokens
      (attention sinks) plus the most recent tokens, evicting the middle.

    The layer returns an `eviction_mask` tensor `[B, S]` (True = keep) during
    the pre-fill step when an eviction policy is active. In subsequent decode
    steps, pass this mask as `attention_eviction_mask` to restrict attention
    to non-evicted positions.

    We use the notation `B`, `T`, `S` below, where `B` is the batch dimension,
    `T` is the target sequence length, and `S` is the source sequence length.
    Note that during generative decoding, `T` is usually 1 (you are
    generating a target sequence of length one to predict the next token).

    Call arguments:
        query: Query `Tensor` of shape `(B, T, dim)`.
        value: Value `Tensor` of shape `(B, S*, dim)`. if `cache` is None`,
            `S*` must equal `S` and match the shape of `attention_mask`. If
            `cache` is not `None`, `S*` can be any length less than `S`, and
            the computed value will be spliced into `cache` at
            `cache_update_index`.
        key: Optional key `Tensor` of shape `(B, S*, dim)`. If `cache` is
            `None`, `S*` must equal `S` and match the shape of
            `attention_mask`. If `cache` is not `None`, `S*` can be any
            length less than `S`, and the computed value will be spliced into
            `cache` at `cache_update_index`.
        attention_mask: a boolean mask of shape `(B, T, S)`. `attention_mask`
            prevents attention to certain positions. The boolean mask specifies
            which query elements can attend to which key elements, 1 indicates
            attention and 0 indicates no attention. Broadcasting can happen for
            the missing batch dimensions and the head dimension.
        cache: a dense float Tensor. The key/value cache, of shape
            `[B, 2, S, num_heads, key_dims]`, where `S` must agree with the
            `attention_mask` shape. This argument is intended for use during
            generation to avoid recomputing intermediate state.
        cache_update_index: a int or int Tensor, the index at which to update
            `cache` (usually the index of the current token being processed
            when running generation). If `cache_update_index=None` while
            `cache` is set, the cache will not be updated.
        attention_eviction_mask: Optional boolean Tensor of shape `[B, S]`.
            When provided (typically during decode steps after pre-fill
            eviction), positions where the mask is False will be excluded from
            attention. This is used together with `kv_budget` /
            `eviction_policy` to enforce the compressed KV cache across all
            decode steps.
        training: a boolean indicating whether the layer should behave in
            training mode or in inference mode.

    Returns:
        When `cache` is None:
            `attention_output`, a Tensor of shape `(B, T, dim)`.
        When `cache` is set and no eviction policy is active:
            `(attention_output, cache)` tuple.
        When `cache` is set and this is a pre-fill step with eviction active:
            `(attention_output, cache, eviction_mask)` tuple, where
            `eviction_mask` is a `[B, S]` boolean Tensor (True = keep).
    """

    def __init__(
        self,
        *args,
        kv_budget=None,
        eviction_policy=None,
        snapkv_window_size=32,
        snapkv_kernel_size=5,
        streaming_llm_n_sink=4,
        **kwargs,
    ):
        """Initialize CachedMultiHeadAttention.

        Args:
            kv_budget: int or None. Maximum number of KV pairs to retain
                after pre-fill eviction. If None, no eviction is applied.
            eviction_policy: str or None. Eviction strategy to use. One of:
                `"h2o"`, `"snapkv"`, `"streaming_llm"`. Requires `kv_budget`.
            snapkv_window_size: int. For `"snapkv"` policy: number of recent
                tokens to use for scoring. Default 32.
            snapkv_kernel_size: int. For `"snapkv"` policy: average-pooling
                kernel size for attention-score smoothing. Default 5.
            streaming_llm_n_sink: int. For `"streaming_llm"` policy: number
                of initial "attention sink" tokens to always retain. Default 4.
            **kwargs: Passed to `keras.layers.MultiHeadAttention`.
        """
        super().__init__(*args, **kwargs)
        if eviction_policy is not None and kv_budget is None:
            raise ValueError(
                "`kv_budget` must be set when `eviction_policy` is not None. "
                f"Got eviction_policy={eviction_policy!r}, kv_budget=None."
            )
        valid_policies = ("h2o", "snapkv", "streaming_llm", None)
        if eviction_policy not in valid_policies:
            raise ValueError(
                f"`eviction_policy` must be one of {valid_policies}. "
                f"Got: {eviction_policy!r}."
            )
        self.kv_budget = kv_budget
        self.eviction_policy = eviction_policy
        self.snapkv_window_size = snapkv_window_size
        self.snapkv_kernel_size = snapkv_kernel_size
        self.streaming_llm_n_sink = streaming_llm_n_sink

    def _compute_attention_based_scores(self, query, key):
        """Compute attention importance scores independently.

        We compute `[B, H, T, S]` attention weights from already-projected
        query and key tensors.  This is used *only* during pre-fill eviction
        so the extra compute is a one-time cost.

        Args:
            query: projected query tensor `[B, T, H, D]` (from _query_dense).
            key: projected key tensor `[B, S, H, D]` (from _key_dense).

        Returns:
            attention_weights `[B, H, T, S]` after softmax.
        """
        # Transpose to [B, H, T, D] and [B, H, S, D]
        q = ops.transpose(query, axes=[0, 2, 1, 3])  # [B, H, T, D]
        k = ops.transpose(key, axes=[0, 2, 1, 3])  # [B, H, S, D]
        head_dim = ops.shape(q)[-1]
        scale = ops.cast(head_dim, "float32") ** (-0.5)
        q_f = ops.cast(q, "float32") * scale
        k_f = ops.cast(k, "float32")
        # Scores [B, H, T, S]
        scores = ops.matmul(q_f, ops.transpose(k_f, axes=[0, 1, 3, 2]))
        # Causal mask (lower triangular)
        T = ops.shape(scores)[-2]
        S = ops.shape(scores)[-1]
        causal = ops.cast(ops.tril(ops.ones([T, S], dtype="bool")), "float32")
        scores = scores + (1.0 - causal) * (-1e9)
        attn_weights = ops.cast(ops.softmax(scores, axis=-1), query.dtype)
        return attn_weights  # [B, H, T, S]

    def _compute_eviction_mask(self, query_proj, key_proj, seq_len):
        """Compute a boolean eviction mask [B, S] (True = keep).

        Args:
            query_proj: projected query tensor `[B, T, H, D]`.
            key_proj: projected key tensor `[B, S, H, D]`.
            seq_len: int, length of the current sequence (S).

        Returns:
            A `[B, S]` boolean Tensor. True where a KV position should be
            retained; False where it should be evicted.
        """
        budget = self.kv_budget
        policy = self.eviction_policy
        bsz = ops.shape(query_proj)[0]

        if policy == "streaming_llm":
            # Purely positional: no attention scores needed.
            n_sink = self.streaming_llm_n_sink
            n_recent = budget - n_sink
            n_evict = seq_len - budget
            sink_scores = ops.ones([bsz, n_sink])
            evict_scores = ops.zeros([bsz, n_evict])
            recent_scores = ops.ones([bsz, n_recent])
            importance = ops.concatenate(
                [sink_scores, evict_scores, recent_scores], axis=1
            )  # [B, S]
        else:
            # H2O and SnapKV need attention weights.
            attn_weights = self._compute_attention_based_scores(
                query_proj, key_proj
            )  # [B, H, T, S]

            if policy == "h2o":
                importance = ops.mean(
                    ops.sum(attn_weights, axis=-2), axis=1
                )  # [B, S]

            elif policy == "snapkv":
                window = min(self.snapkv_window_size, seq_len - 1)
                # Window attention: last `window` queries, first (S-window) keys
                window_attn = attn_weights[
                    :, :, -window:, :-window
                ]  # [B, H, win, S-win]
                scores = ops.mean(window_attn, axis=2)  # [B, H, S-win]

                # Avg-pool smoothing (backend-agnostic manual implementation)
                k = self.snapkv_kernel_size
                pad = k // 2
                n_heads = ops.shape(scores)[1]
                s_main = seq_len - window

                scores_3d = ops.reshape(scores, [bsz * n_heads, s_main, 1])
                padded = ops.pad(scores_3d, [[0, 0], [pad, pad], [0, 0]])
                pooled_parts = [
                    ops.slice(padded, [0, i, 0], [bsz * n_heads, s_main, 1])
                    for i in range(k)
                ]
                pooled = ops.mean(ops.stack(pooled_parts, axis=-1), axis=-1)
                scores = ops.reshape(pooled, [bsz, n_heads, s_main])

                # Pad window positions back with max score so they're kept
                max_score = ops.max(scores) + 1.0
                window_scores = ops.full(
                    [bsz, n_heads, window], max_score, dtype=scores.dtype
                )
                joint = ops.concatenate(
                    [scores, window_scores], axis=-1
                )  # [B, H, S]
                importance = ops.mean(joint, axis=1)  # [B, S]

        # Keep top-kv_budget positions.
        # ops.top_k returns (values, indices); use [0] for backend compat.
        topk_vals = ops.top_k(ops.cast(importance, "float32"), k=budget)[
            0
        ]  # [B, budget]
        threshold = topk_vals[:, -1:]  # [B, 1]
        eviction_mask = ops.cast(
            ops.cast(importance, "float32") >= threshold, "bool"
        )  # [B, S]
        return eviction_mask

    def _apply_eviction_to_attention_mask(
        self, attention_mask, eviction_mask, seq_len
    ):
        """Merge a [B, S] eviction mask into a [B, T, S] attention mask.

        Args:
            attention_mask: `[B, T, S]` or `[B, 1, 1, S]` bool Tensor, or
                None.
            eviction_mask: `[B, S]` bool Tensor (True = keep).
            seq_len: int, sequence length S.

        Returns:
            Updated attention mask (same shape as input, or `[B, 1, S]` if
            `attention_mask` was None).
        """
        # Shape eviction_mask → [B, 1, S] for broadcasting over T.
        eviction_mask_3d = ops.expand_dims(eviction_mask, axis=1)  # [B, 1, S]
        if attention_mask is None:
            return eviction_mask_3d
        # Cast to same dtype for minimum (0/1 logic).
        attn = ops.cast(attention_mask, "bool")
        evic = ops.cast(eviction_mask_3d, "bool")
        return ops.logical_and(attn, evic)

    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        cache=None,
        cache_update_index=None,
        attention_eviction_mask=None,
        training=None,
    ):
        if key is None:
            key = value

        query = self._query_dense(query)

        # If cache is not `None`, we will use the cache to compute the final
        # key and value tensors. If `cache_update_index` is not None, we will
        # first update the cache before use. `cache = None` handles the
        # training case.
        if cache is not None:
            key_cache = cache[:, 0, ...]
            value_cache = cache[:, 1, ...]
            if cache_update_index is None:
                key = key_cache
                value = value_cache
            else:
                key_update = self._key_dense(key)
                value_update = self._value_dense(value)
                start = [0, cache_update_index, 0, 0]
                key = ops.slice_update(key_cache, start, key_update)
                value = ops.slice_update(value_cache, start, value_update)
                cache = ops.stack((key, value), axis=1)
        else:
            if cache_update_index is not None:
                raise ValueError(
                    "`cache_update_index` should not be set if `cache` is "
                    f"`None`. Received: cache={cache}, "
                    f"cache_update_index={cache_update_index}"
                )
            key = self._key_dense(key)
            value = self._value_dense(value)

        # Merge any existing eviction mask into the attention mask so evicted
        # positions are excluded during decode steps.
        if attention_eviction_mask is not None:
            seq_len = ops.shape(key)[1]
            attention_mask = self._apply_eviction_to_attention_mask(
                attention_mask, attention_eviction_mask, seq_len
            )

        attention_output, _ = self._compute_attention(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            training=training,
        )
        attention_output = self._output_dense(attention_output)

        # KV cache compression — applied once after the pre-fill pass.
        # Pre-fill is detected by: cache is set, cache_update_index == 0, and
        # more than one token was processed (T > 1, i.e. not a decode step).
        eviction_mask = None
        if (
            cache is not None
            and self.eviction_policy is not None
            and cache_update_index is not None
            and cache_update_index == 0
            and ops.shape(query)[1] > 1  # T > 1 → this is a pre-fill
        ):
            seq_len = ops.shape(key)[1]
            if seq_len > self.kv_budget:
                # query/key are already in projected head form [B, T/S, H, D]
                eviction_mask = self._compute_eviction_mask(query, key, seq_len)

        if cache is not None:
            if eviction_mask is not None:
                return attention_output, cache, eviction_mask
            return attention_output, cache
        return attention_output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kv_budget": self.kv_budget,
                "eviction_policy": self.eviction_policy,
                "snapkv_window_size": self.snapkv_window_size,
                "snapkv_kernel_size": self.snapkv_kernel_size,
                "streaming_llm_n_sink": self.streaming_llm_n_sink,
            }
        )
        return config
