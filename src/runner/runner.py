import numpy as np
import tiktoken
from loader.load_safetensor import load_safetensors, extract_tensor_bytes
from model.gpt_model import GPTModel

if __name__ == "__main__":
    MAX_TOKENS = 50

    FILE_PATH = "C:\\Users\\youra\\models\\gpt2\\model.safetensors"

    header_size, header, raw = load_safetensors(FILE_PATH)
    tensor_map = extract_tensor_bytes(header, raw)
    gpt_model = GPTModel(header, tensor_map)
    tokenizer = tiktoken.get_encoding("gpt2")

    while True:
        context_ids = []
        user_message = input("> ")

        if user_message == "/quit":
            break

        token_ids = tokenizer.encode(user_message)

        context_ids.extend(token_ids)

        for i in range(MAX_TOKENS):
            logits = gpt_model.forward(context_ids)
            last_row = logits[-1]
            next_token_id = np.argmax(last_row)
            context_ids.extend([next_token_id])
            decoded = tokenizer.decode([next_token_id])

            print(decoded, end="", flush=True)

        print()
