#!/usr/bin/env python3
"""Parse a locally downloaded GGUF header prefix: metadata KV + tensor names/dims.

Used to inspect the MTP draft head (mtp-Qwen3.8-Flash-Next-Q8_0.gguf) without
fetching the whole 4 GB file: only the first few MB are needed for KV metadata and
the tensor directory.  Answers, for the #29811 MTP question:
  - does the draft carry its own attention.compress_ratios array?  (If it did,
    editing it would be a no-rebuild way to change hparams.indexer_kpool, which is
    the value that makes graph_mtp build the k-pool input that then has no backend
    buffer — qwen4exp.cpp:571 + :697 at b11430.)
  - what architecture / tensor set does it declare (dense attention? own lm_head?)
Read-only; local file parse.
"""
import struct
import sys


def rd_str(b, p):
    (ln,) = struct.unpack_from("<Q", b, p); p += 8
    return b[p:p + ln].decode("utf-8", "replace"), p + ln


SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
          4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
          10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


def rd_val(b, p, vt, depth=0):
    if vt in SCALAR:
        f, n = SCALAR[vt]
        return struct.unpack_from(f, b, p)[0], p + n
    if vt == 7:
        return bool(b[p]), p + 1
    if vt == 8:
        return rd_str(b, p)
    if vt == 9:
        (et,) = struct.unpack_from("<I", b, p); p += 4
        (cnt,) = struct.unpack_from("<Q", b, p); p += 8
        vals = []
        for _ in range(cnt):
            v, p = rd_val(b, p, et, depth + 1)
            vals.append(v)
        return vals, p
    raise ValueError(f"type {vt} at {p}")


def main():
    path = sys.argv[1]
    subs = [s.lower() for s in sys.argv[2:]]
    with open(path, "rb") as f:
        b = f.read()
    assert b[:4] == b"GGUF", "not GGUF"
    (ver,) = struct.unpack_from("<I", b, 4)
    (n_tensors,) = struct.unpack_from("<Q", b, 8)
    (n_kv,) = struct.unpack_from("<Q", b, 16)
    print(f"GGUF v{ver}  n_tensors={n_tensors}  n_kv={n_kv}  file_prefix={len(b)} bytes")
    p = 24
    print("--- metadata KV ---")
    if n_kv == 0:
        print("  (none: the file carries NO metadata keys)")
    for _ in range(n_kv):
        k, p = rd_str(b, p)
        (vt,) = struct.unpack_from("<I", b, p); p += 4
        v, p = rd_val(b, p, vt)
        if not subs or any(s in k.lower() for s in subs):
            vs = str(v)
            print(f"  {k} = {vs[:300]}{' ...' if len(vs) > 300 else ''}")
    print("--- tensor directory ---")
    for i in range(n_tensors):
        name, p = rd_str(b, p)
        (nd,) = struct.unpack_from("<I", b, p); p += 4
        dims = struct.unpack_from(f"<{nd}Q", b, p); p += 8 * nd
        (dt,) = struct.unpack_from("<I", b, p); p += 4
        (off,) = struct.unpack_from("<Q", b, p); p += 8
        print(f"  [{i:2}] {name:55s} dt={dt:3d} dims={list(dims)}")


if __name__ == "__main__":
    sys.exit(main())
