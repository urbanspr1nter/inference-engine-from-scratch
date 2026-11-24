import numpy as np


class Attention:
    def __init__(
        self,
        c_attn_w: np.ndarray,
        c_attn_b: np.ndarray,
        c_proj_w: np.ndarray,
        c_proj_b: np.ndarray,
        n_heads: int,
    ):
        self._c_attn_w = c_attn_w
        self._c_attn_b = c_attn_b
        self._c_proj_w = c_proj_w
        self._c_proj_b = c_proj_b

        self._n_heads = n_heads

    def forward(self, x: np.ndarray):
        seq_len, d_model = x.shape

        Wq = self._c_attn_w[:, :d_model]
        Wk = self._c_attn_w[:, d_model : 2 * d_model]
        Wv = self._c_attn_w[:, 2 * d_model :]

        bq = self._c_attn_b[:d_model]
        bk = self._c_attn_b[d_model : 2 * d_model]
        bv = self._c_attn_b[2 * d_model :]

        q = x @ Wq + bq
        k = x @ Wk + bk
        v = x @ Wv + bv

        d_head = d_model // self._n_heads

        q_h = q.reshape(seq_len, self._n_heads, d_head).transpose(
            1, 0, 2
        )  # (12, seq, 64)
        k_h = k.reshape(seq_len, self._n_heads, d_head).transpose(1, 0, 2)
        scores = q_h @ k_h.transpose(0, 2, 1) / (d_head**0.5)  # (12, seq, seq)

        allowed = np.tril(np.ones((len(x), len(x)), dtype=bool))
        masked = np.where(allowed, scores, -1e10)

        # softmax
        e = np.exp(masked - masked.max(-1, keepdims=True))
        w = e / e.sum(-1, keepdims=True)

        v_h = v.reshape(seq_len, self._n_heads, d_head).transpose(
            1, 0, 2
        )  # (12, seq, 64)

        out_h = w @ v_h

        out = out_h.transpose(1, 0, 2).reshape(
            seq_len, d_model
        )  # (seq, 768) — concat heads

        return out @ self._c_proj_w + self._c_proj_b
