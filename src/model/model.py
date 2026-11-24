import numpy as np
from loader.load_safetensor import as_np_buffer
from model.token_embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from model.block import Block
from model.layer_norm import LayerNorm

N_HEADS = 12
N_LAYERS = 12  # block count -- its own config value (GPT-2 medium: 16 heads, 24 layers)


class GPTModel:
    def __init__(self, header: dict, tensor_map: dict[str, memoryview]):
        self._header = header
        self._tensor_map = tensor_map

        self._wte_weight = as_np_buffer("wte.weight", self._header, self._tensor_map)

        self._token_embedding = TokenEmbedding(self._wte_weight)

        self._wpe_weight = as_np_buffer("wpe.weight", self._header, self._tensor_map)
        self._positional_embedding = PositionalEmbedding(self._wpe_weight)

        self._blocks = [
            Block(i, N_HEADS, self._header, self._tensor_map) for i in range(N_LAYERS)
        ]

        self._layer_norm_final = LayerNorm(
            as_np_buffer("ln_f.weight", self._header, self._tensor_map),
            as_np_buffer("ln_f.bias", self._header, self._tensor_map),
        )

    def forward(self, token_ids: np.ndarray):
        token_embeddings = self._token_embedding.forward(token_ids)

        positions = np.arange(len(token_ids))
        positional_embeddings = self._positional_embedding.forward(positions)

        h = token_embeddings + positional_embeddings

        for block in self._blocks:
            h = block.forward(h)

        h = self._layer_norm_final.forward(h)

        return h @ self._wte_weight.T
