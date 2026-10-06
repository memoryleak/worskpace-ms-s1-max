#!/usr/bin/env python3
"""Range-read a GGUF header (metadata only) from a URL and print selected KV keys.

Why: the #29811 MTP startup assert is decided by `hparams.indexer_kpool`, which
qwen4exp load_hparams derives from the GGUF `attention.compress_ratios` array
(src/models/qwen4exp.cpp:62-74 at b11430): it is the single non-zero compress ratio
across all layers, and load throws if the array is absent-with-no-QSA or if the
ratio does not divide indexer_top_k.  graph_mtp then builds a k-pool input whenever
`indexer_kpool > 0` (qwen4exp.cpp:571) — including for the dense-attention draft
block, which is the bug.

If the *draft* GGUF's own metadata carries a compress-ratio array, that is a
potential no-rebuild lever (edit metadata with gguf-set-metadata, no retrain, no
re-quantise).  This reads only the header via HTTP Range so a 4 GB file costs a few
hundred KB.  Read-only w.r.t. the inference box; this talks to huggingface.co.

Usage: gguf_header_peek.py <url> [key-substring ...]
"""
import struct
import sys
import urllib.request

GGUF_MAGIC = b"GGUF"
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
SCALAR = {T_UINT8: (1, "<B"), T_INT8: (1, "<b"), T_UINT16: (2, "<H"), T_INT16: (2, "<h"),
          T_UINT32: (4, "<I"), T_INT32: (4, "<i"), T_FLOAT32: (4, "<f"),
          T_UINT64: (8, "<Q"), T_INT64: (8, "<q"), T_FLOAT64: (8, "<d")}


class Reader:
    """Sequential reader over a bytes buffer, refilled by ranged HTTP GETs."""

    def __init__(self, url, chunk=1 << 20):
        self.url = url
        self.chunk = chunk
        self.buf = b""
        self.pos = 0
        self.eof_at = None

    def need(self, n):
        while self.pos + n > len(self.buf):
            if self.eof_at is not None and len(self.buf) >= self.eof_at:
                raise EOFError("out of data")
            start = len(self.buf)
            end = max(start + self.chunk, self.pos + n)
            req = urllib.request.Request(
                self.url, headers={"Range": f"bytes={start}-{end}",
                                   "User-Agent": "gguf-peek/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if not data:
                self.eof_at = len(self.buf)
                raise EOFError("server returned no more data")
            self.buf += data

    def read(self, n):
        self.need(n)
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(n))[0]


def read_string(r):
    ln = r.unpack("<Q")
    return r.read(ln).decode("utf-8", "replace")


def read_val(r, vtype):
    if vtype in SCALAR:
        return r.unpack(SCALAR[vtype][1])
    if vtype == T_BOOL:
        return bool(r.unpack("<B"))
    if vtype == T_STRING:
        return read_string(r)
    if vtype == T_ARRAY:
        elem = r.unpack("<I")
        cnt = r.unpack("<Q")
        if cnt > 4096:
            # do not pull huge arrays; report only
            return f"<array {elem} x {cnt} (not read)>"
        return [read_val(r, elem) for _ in range(cnt)]
    raise ValueError(f"unknown type {vtype}")


def main():
    url = sys.argv[1]
    subs = [s.lower() for s in sys.argv[2:]]
    r = Reader(url)
    if r.read(4) != GGUF_MAGIC:
        print("not a GGUF")
        return 1
    ver = r.unpack("<I")
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    print(f"GGUF v{ver}  n_tensors={n_tensors}  n_kv={n_kv}")
    hits = {}
    for _ in range(n_kv):
        k = read_string(r)
        vt = r.unpack("<I")
        v = read_val(r, vt)
        low = k.lower()
        if not subs or any(s in low for s in subs):
            hits[k] = (vt, v)
        if vt == T_ARRAY and isinstance(v, list) and len(v) > 64:
            hits.setdefault(k, (vt, f"<array len {len(v)}>"))
    for k, (vt, v) in sorted(hits.items()):
        print(f"  {k}  (type {vt}) = {v}")
    if not subs:
        print(f"  ... header bytes consumed: {r.pos}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
