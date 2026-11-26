import numpy as np
from loader.load_safetensor import as_np_buffer
from model.layer_norm import LayerNorm
from model.mlp import Mlp
from model.attention import Attention


class Block:
    def __init__(
        self, i: int, n_heads: int, header: dict, tensor_map: dict[str, memoryview]
    ):
        self._header = header
        self._tensor_map = tensor_map

        tensor_names = []
        prefix = f"h.{i}"
        for k, _ in header.items():
            if k.startswith(f"{prefix}."):
                tensor_names.append(k)

        print(tensor_names)

        for name in tensor_names:
            if name not in self._tensor_map:
                raise KeyError(f"{name} is not found in the tensor map!!!")

        # We have everything we need

        # Build LayerNorm
        self._layer_norm_1 = LayerNorm(
            as_np_buffer(f"{prefix}.ln_1.weight", self._header, self._tensor_map),
            as_np_buffer(f"{prefix}.ln_1.bias", self._header, self._tensor_map),
        )
        self._layer_norm_2 = LayerNorm(
            as_np_buffer(f"{prefix}.ln_2.weight", self._header, self._tensor_map),
            as_np_buffer(f"{prefix}.ln_2.bias", self._header, self._tensor_map),
        )

        # Build Mlp
        self._mlp = Mlp(
            as_np_buffer(f"{prefix}.mlp.c_fc.weight", self._header, self._tensor_map),
            as_np_buffer(f"{prefix}.mlp.c_fc.bias", self._header, self._tensor_map),
            as_np_buffer(f"{prefix}.mlp.c_proj.weight", self._header, self._tensor_map),
            as_np_buffer(f"{prefix}.mlp.c_proj.bias", self._header, self._tensor_map),
        )

        # Build Attention
        self._attention = Attention(
            as_np_buffer(
                f"{prefix}.attn.c_attn.weight", self._header, self._tensor_map
            ),
            as_np_buffer(f"{prefix}.attn.c_attn.bias", self._header, self._tensor_map),
            as_np_buffer(
                f"{prefix}.attn.c_proj.weight", self._header, self._tensor_map
            ),
            as_np_buffer(f"{prefix}.attn.c_proj.bias", self._header, self._tensor_map),
            n_heads,
        )

    def forward(self, x: np.ndarray):
        x = x + self._attention.forward(self._layer_norm_1.forward(x))
        x = x + self._mlp.forward(self._layer_norm_2.forward(x))

        return x
