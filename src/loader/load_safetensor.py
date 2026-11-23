import os
import json
import struct
import numpy as np

FILE_PATH = "C:\\Users\\youra\\models\\gpt2\\model.safetensors"

NUMPY_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load_safetensors(file_path: str) -> tuple[any, any]:
    path_exists = os.path.exists(file_path)

    if not path_exists:
        raise IOError(f"Path does not exist: {file_path}")

    with open(file_path, "rb") as file:
        (n,) = struct.unpack("<Q", file.read(8))
        header = json.loads(file.read(n))
        raw = memoryview(file.read())

    return n, header, raw


def extract_tensor_bytes(header: dict, raw_buffer: memoryview) -> dict[str, memoryview]:
    """Extracts the tensors based on header data (name and offsets) and
    returns a dict of tensor name -> view of bytes for easy reference.
    """
    ignore_tensor_names = set(["__metadata__"])
    result = {}
    for k, v in header.items():
        if k in ignore_tensor_names:
            continue

        tensor_name = k
        data_offsets = v["data_offsets"]
        begin_offset = data_offsets[0]
        end_offset = data_offsets[1]

        view_bytes = raw_buffer[begin_offset:end_offset]

        result[tensor_name] = view_bytes

    return result


def get_shape(header: dict, tensor_name: str):
    if tensor_name not in header:
        raise ValueError(f"{tensor_name} not in header!")

    shape = header[tensor_name]["shape"]

    return shape


def get_dtype(header: dict, tensor_name: str):
    if tensor_name not in header:
        raise ValueError(f"{tensor_name} not in header!")

    dtype = header[tensor_name]["dtype"]  # the string from the JSON, e.g. "F32"
    if dtype not in NUMPY_DTYPES:
        raise ValueError(f"{tensor_name}: unsupported dtype {dtype!r}")

    return NUMPY_DTYPES[dtype]


def as_np_buffer(tensor_name: str, header: dict, tensor_map: dict):
    if tensor_name not in header:
        raise ValueError(f"{tensor_name} not in header!")

    dtype = get_dtype(header, tensor_name)

    np_buffer = np.frombuffer(tensor_map[tensor_name], dtype=dtype).reshape(
        get_shape(header, tensor_name)
    )

    return np_buffer


if __name__ == "__main__":
    header_size, header, raw = load_safetensors(FILE_PATH)
    output = {"header_size": header_size, "header": header}

    # Print out the header size, and header
    # print(json.dumps(output, indent=2))

    tensor_names = [k for k, v in header.items() if "data_offsets" in v]
    print(f"We have {len(tensor_names)} tensors")
    print("curr tensor name", tensor_names[1])

    tensor_name = tensor_names[1]
    print(header[tensor_name])

    # Has a key->value pair of tensor names to the bytes
    tensor_map = extract_tensor_bytes(header, raw)

    wte_np_buff = as_np_buffer("wte.weight", header, tensor_map)
    print(wte_np_buff)

    wpe_np_buff = as_np_buffer("wpe.weight", header, tensor_map)
    print(wpe_np_buff)
