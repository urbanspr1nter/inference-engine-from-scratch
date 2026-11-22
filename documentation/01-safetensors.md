# Safetensors Spec

- 8 bytes: `N`, an unsigned little-endian 64-bit integer, containing the size of the header
- N bytes: a JSON UTF-8 string representing the header.
  - The header data MUST begin with a `{` character (0x7B).
  - The header data MAY be trailing padded with whitespace (0x20).
  - The header is a dict like `{"TENSOR_NAME": {"dtype": "F16", "shape": [1, 16, 256], "data_offsets": [BEGIN, END]}, "NEXT_TENSOR_NAME": {...}, ...}`,
    - `data_offsets` point to the tensor data relative to the beginning of the byte buffer (i.e. not an absolute position in the file),
      with `BEGIN` as the starting offset and `END` as the one-past offset (so total tensor byte size = `END - BEGIN`).
  - A special key `__metadata__` is allowed to contain free form string-to-string map. Arbitrary JSON is not allowed, all values must be strings.
- Rest of the file: byte-buffer.

Notes:
 - Duplicate keys are disallowed. Not all parsers may respect this.
 - This library writes `__metadata__` keys in sorted order; readers must not rely on key order.
 - In general the subset of JSON is implicitly decided by `serde_json` for
   this library. Anything obscure might be modified at a later time, that odd ways
   to represent integer, newlines and escapes in utf-8 strings. This would only
   be done for safety concerns
 - Tensor values are not checked against, in particular NaN and +/-Inf could
   be in the file
 - Empty tensors (tensors with 1 dimension being 0) are allowed.
   They are not storing any data in the databuffer, yet retaining size in the header.
   They don't really bring a lot of values but are accepted since they are valid tensors
   from traditional tensor libraries perspective (torch, tensorflow, numpy, ..).
 - 0-rank Tensors (tensors with shape `[]`) are allowed, they are merely a scalar.
 - The byte buffer needs to be entirely indexed, and cannot contain holes. This prevents
   the creation of polyglot files.
 - Endianness: Little-endian.
   moment.
 - Order: 'C' or row-major.
 - Notes: Some smaller than 1 byte dtypes appeared, which make alignment tricky. Non traditional APIs might be required for those.


# FAQ

Good questions — two of the three trip people up.

## 1. "Byte-buffer" = everything after the header

Yes. The file is exactly two regions:

| Region | Absolute file offsets | Size |
|---|---|---|
| header length `N` | `0 .. 7` | 8 bytes, u64 LE |
| header JSON | `8 .. 8+N-1` | `N` bytes |
| **byte-buffer** | `8+N .. filesize-1` | `filesize - 8 - N` |

The header length prefix is *not* part of the buffer, and neither is the whitespace padding that trails the JSON (that padding is inside the `N` bytes — it exists so `8+N` lands on an 8-byte boundary, which the reference implementation ensures for aligned access; the 8-byte u64 prefix is why padding to a multiple of 8 works out). The buffer can legitimately be empty (a header of `{}`).

The point of the name: it's an unstructured blob. The header is the only thing that tells you what's in it.

## 2. `BEGIN`/`END` are relative to the buffer, not the file

Your guess is wrong, and it's the most common misreading. `BEGIN` is **not** a file position and not "the byte after the header" — it's a 0-based offset *inside* the byte-buffer. Typically the first tensor has `BEGIN == 0`, meaning its data starts at the header's end.

```
absolute_offset = 8 + N + data_offset
tensor_bytes    = file[8+N+BEGIN : 8+N+END]      # END is one-past, so END-BEGIN = size in bytes
```

Worked example, `N = 128`:

| Thing | Header value | Absolute file bytes |
|---|---|---|
| tensor `a` (`F32`, `[2,2]`, 16 B) | `[0, 16]` | `136 .. 151` |
| tensor `b` (`F16`, `[3]`, 6 B) | `[16, 22]` | `152 .. 157` |
| buffer | — | `136 .. 157` (22 bytes) |
| whole file | — | 158 bytes |

Also note: offsets are **byte** offsets, not element indices — `F16` with `END-BEGIN == 6` is 3 elements. And in files written by the reference implementation the tensors are concatenated so that, sorted by `BEGIN`, the ranges tile the buffer with no gaps (`END` of one equals `BEGIN` of the next). Treat that as a convention to sanity-check against, not something to hard-require when reading.

## 3. `__metadata__`

It's an ordinary top-level key whose value is a flat `string → string` object (no nested objects, no numbers, no booleans, no nulls — anything non-string makes the file invalid). Real Hugging Face output looks like:

```json
{
  "__metadata__": {
    "format": "pt",
    "author": "youra",
    "my_custom_key": "anything you like, as long as it is a string"
  },
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [768, 768],
    "data_offsets": [0, 1179648]
  }
}
```

`{"format": "pt"}` (or `"tf"`, `"flax"`, `"mlx"`) is the one value the ecosystem actually depends on — it tells the loader which framework tensors to build. Numbers must be quoted: `{"num_layers": "32"}`, never `32`. The block is optional; when present the reference writer puts it first, and `__metadata__` is a reserved name, so no tensor may be called that.

## Reader sketch

```python
import json, struct
import numpy as np

DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64, "I32": np.int32,
          "U8": np.uint8, "BOOL": np.bool_}

def read_safetensors(raw: bytes):
    n, = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8+n])          # trailing 0x20 is legal JSON whitespace
    meta = header.pop("__metadata__", {})
    buf = memoryview(raw)[8+n:]              # offset 0 here == absolute 8+n

    out = {}
    for name, t in header.items():
        begin, end = t["data_offsets"]
        assert 0 <= begin <= end <= len(buf), name        # never trust the file
        nbytes = end - begin
        assert int(np.prod(t["shape"], dtype=np.int64)) * (
            nbytes // max(1, int(np.prod(t["shape"], dtype=np.int64)))) == nbytes, name
        out[name] = np.frombuffer(buf[begin:end], dtype=DTYPES[t["dtype"]]).reshape(t["shape"])
    return out, meta
```

Three practical notes for your engine: `BF16` has no numpy dtype (view as `uint16` and reinterpret on read), `N` counts UTF-8 **bytes** so a non-ASCII tensor name makes it differ from the character count, and for real models memmap the file once and hand `np.frombuffer(mm, offset=8+N+begin, count=...)` out per tensor — that's the whole reason offsets are buffer-relative.

Sources: [safetensors README](https://github.com/safetensors/safetensors/blob/main/README.md), [file format / alignment notes](https://deepwiki.com/huggingface/safetensors/2.1-file-format).

## 4. Little-endian

Least significant byte (LSB) is read first.