import os
import json
import struct

FILE_PATH = "C:\\Users\\youra\\models\\gpt2\\model.safetensors"


def load_safetensors(file_path: str) -> tuple[any, any]:
    path_exists = os.path.exists(file_path)

    if not path_exists:
        raise IOError(f"Path does not exist: {file_path}")

    with open(file_path, "rb") as file:
        (n,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(n))
        raw = memoryview(file.read())

    return n, header, raw


header_size, header, raw = load_safetensors(FILE_PATH)
output = {"header_size": header_size, "header": header}

# Print out the header size, and header
# print(json.dumps(output, indent=2))

tensor_names = [k for k, v in header.items() if "data_offsets" in v]
print(f"We have {len(tensor_names)} tensors")
print("curr tensor name", tensor_names[1])

tensor_name = tensor_names[1]
print(header[tensor_name])

# data_offsets BEGIN (inclusive), END (exclusive)
data_offsets = header[tensor_name]["data_offsets"]
start_offset, end_offset = data_offsets[0], data_offsets[1]
data = raw[start_offset:end_offset]

materialized = data.tobytes()
print(struct.unpack("<ffff", data[:16]))
