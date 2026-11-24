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

        Q = x @ Wq + bq
        K = x @ Wk + bk
        V = x @ Wv + bv

        # scaling factor
        d_head = d_model // self._n_heads

        # mask
        allowed_mask = np.tril(np.ones((len(x), len(x)), dtype=bool))

        q_h = Q.reshape(seq_len, self._n_heads, d_head).transpose(1, 0, 2)
        k_h = K.reshape(seq_len, self._n_heads, d_head).transpose(1, 0, 2)
        v_h = V.reshape(seq_len, self._n_heads, d_head).transpose(1, 0, 2)

        scores = q_h @ k_h.transpose(0, 2, 1) / (d_head**0.5)

        masked = np.where(allowed_mask, scores, -1e10)

        e = np.exp(masked - masked.max(axis=-1, keepdims=True))

        out = e / e.sum(axis=-1, keepdims=True)

        out_h = out @ v_h

        h = out_h.transpose(1, 0, 2).reshape(seq_len, d_model)

        return h @ self._c_proj_w + self._c_proj_b
