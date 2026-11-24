import numpy as np
from model.token_embedding import TokenEmbedding
from model.positional_embedding import PositionalEmbedding
from loader.load_safetensor import as_np_buffer


class GPTModel:
    def __init__(self, header: dict, tensor_map: dict[str, memoryview]):
        self._header = header
        self._tensor_map = tensor_map

        self._wte_weight = as_np_buffer("wte.weight", self._header, self._tensor_map)
        self._wpe_weight = as_np_buffer("wpe.weight", self._header, self._tensor_map)

        self._token_embedding = TokenEmbedding(self._wte_weight)
        self._positional_embedding = PositionalEmbedding(self._wpe_weight)

    def forward(self, token_ids: np.ndarray):
        """Performs the forward pass through the entire model architecture of GPT-2"""
        token_embeddings = self._token_embedding.forward(token_ids)

        positions = np.arange(len(token_ids))
        positional_embeddings = self._positional_embedding.forward(positions)

        h = token_embeddings + positional_embeddings

        # FIXME: This will eventually be the logits
        return h
