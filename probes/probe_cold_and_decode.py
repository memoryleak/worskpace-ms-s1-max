#!/usr/bin/env python3
"""Fresh, read-only probe of the live server (Fix #2 / b11391 state).

Two measurements the perf doc does not yet have under the current config:

  1. Genuinely-cold prefill ladder.  benchmark.py reuses fixed per-rung
     seeds (4000+rung), so a second run collides with text already in the
     prefix cache and the "cold" wall times come back as cache hits.  This
     probe salts the seed with the wall clock so every rung is new text.

  2. Decode rate as a function of prompt/context length.  Same streamed
     greedy 200-token decode, but the prompt carries an 8 / 13k / 40k /
     80k-token prefix.  With swa-checkpoints 8 the per-token attention span
     is bounded, so a flat decode-vs-ctx curve is the default expectation;
     a slope would show KV/attention read work competing with weight reads.

Uses only the stdlib.  Request shapes mirror benchmark.py / the perf doc
(stream for decode, non-stream max_tokens=4 for prefill; enable_thinking false,
temperature 0).  All numbers logged to stdout and JSON.

Read-only: only HTTP GET/POST to http://ms-s1-max-01:9931.
"""
import argparse
import json
import random
import sys
import time
import urllib.request

URL = "http://ms-s1-max-01:9931"
MODEL = "qwen3.8-flash-next"
TOKENS_PER_PARA = 135
DECODE_TOKENS = 200
DECODE_CMD = "Count from 1 to 80 separated by spaces."

WORDS = ("the quick brown fox jumps over lazy dogs while data flows through "
         "distributed systems and kernels schedule workloads across "
         "heterogeneous accelerators in modern clusters").split()


def make_prefix(seed, n_paras):
    rng = random.Random(seed)
    lines = []
    for i in range(n_paras):
        line = " ".join(rng.choice(WORDS) for _ in range(120))
        lines.append(f"{line} {i}")
    return "\n".join(lines)


def post(path, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        URL + path, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def chat(messages, max_tokens, stream, timeout):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", payload, timeout)
    if not stream:
        obj = json.loads(resp.read().decode())
        return time.perf_counter() - t0, obj, None

    ttft = None
    n_tok = 0
    last = None
    buf = ""
    for raw in resp:
        buf += raw.decode("utf-8", "replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    n_tok += len(content)
            last = obj
    return time.perf_counter() - t0, last, ttft


def timings_of(obj):
    t = obj.get("timings") or {}
    return (t.get("predicted_per_second"),
            t.get("prompt_ms"), t.get("prompt_per_second"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", type=int, nargs="+",
                    default=[12000, 49000, 122000])
    ap.add_argument("--decode-ctx", type=int, nargs="+",
                    default=[0, 13000, 40000, 80000])
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    salt = int(time.time())
    out = {"url": URL, "model": MODEL,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "salt": salt, "cold_prefill": [], "decode_vs_ctx": []}
    print(f"probe_cold_and_decode  salt={salt}  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    # ---- 1. genuinely cold prefill ladder (salted seeds) ----
    for rung in args.ladder:
        n_paras = max(1, round(rung / TOKENS_PER_PARA))
        text = make_prefix(salt + rung, n_paras)
        wall, obj, _ = chat(
            [{"role": "user", "content": "Reply READY.\n\n" + text}],
            max_tokens=4, stream=False, timeout=args.timeout)
        pt = (obj.get("usage") or {}).get("prompt_tokens", 0)
        _, p_ms, p_pps = timings_of(obj)
        row = {"target": rung, "prompt_tokens": pt, "wall_s": round(wall, 3),
               "server_prompt_ms": p_ms, "server_pps": p_pps}
        out["cold_prefill"].append(row)
        print(f"  prefill {rung:>6} -> {pt:>6} tok  wall {wall:7.2f}s  "
              f"{pt/wall:.0f} tok/s (server {p_pps})", flush=True)

    # ---- 2. decode vs context length (each prefill in this block is also
    #         a cold data point above only for the ladder targets; here the
    #         prefixes are fresh/salted too) ----
    for target in args.decode_ctx:
        if target == 0:
            prompt = DECODE_CMD
        else:
            n_paras = max(1, round(target / TOKENS_PER_PARA))
            prompt = (make_prefix(salt + 9000 + target, n_paras)
                      + "\n" + DECODE_CMD)
        wall, obj, ttft = chat(
            [{"role": "user", "content": prompt}],
            max_tokens=DECODE_TOKENS, stream=True, timeout=args.timeout)
        used = (obj.get("usage") or {}).get("prompt_tokens", 0)
        comp = (obj.get("usage") or {}).get("completion_tokens", 0)
        pred_pps, _, _ = timings_of(obj)
        decode_wall = wall - (ttft or 0.0)
        rate = comp / decode_wall if decode_wall else 0.0
        row = {"target": target, "prompt_tokens": used,
               "completion_tokens": comp, "ttft_s": round(ttft or 0.0, 3),
               "decode_wall_s": round(decode_wall, 3),
               "decode_tok_s": round(rate, 2), "server_pps": pred_pps}
        out["decode_vs_ctx"].append(row)
        print(f"  decode ctx {used:>6} tok -> {comp} tok in {decode_wall:5.2f}s "
              f"= {rate:5.2f} tok/s (server {pred_pps})", flush=True)

    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())