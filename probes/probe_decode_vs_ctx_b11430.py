#!/usr/bin/env python3
"""Decode-rate vs context length under b11430 (sparse-FA discriminator).

Why this probe exists
---------------------
perf doc §4.1 measured decode as a function of context on b11391/b11399 and got a
slope: 28.6 (short) -> 27.4 (13k) -> 26.1 (40k) -> 24.3 (80k) tok/s, i.e. ~15%
decay, recorded as "context/QSA-indexer reads do compete for bandwidth".

Between b11399 and b11430 upstream merged ggml-org/llama.cpp#29639 ("vulkan: sparse
flash attention for quantized K/V", merged 2026-10-05 08:37Z). Reported-by-source:
with a q8_0 KV cache, Flash-Next ran DENSE attention over the whole context because
Vulkan's sparse FA only activated for f16 K/V; #29639 opens the gate for quantized
K/V, with a separate 16x context/kept threshold, and reports +15.6% decode at 64k
and +4.1% at 128k, ~0 below 32k (source: PR body, RDNA3/RDNA4 discrete cards, not
our box).

The gate in ggml-vulkan.cpp at 8345f3339 (b11430) is:

    min_ratio = path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16)
    use_sparse = ... (kv_f16 || path != FA_COOPMAT2) && nem0 == KV
                 && KV >= max(4096, min_ratio * n_kv_max) && ...

so for OUR config (q8_0 KV, so kv_f16 false) sparse FA activates only if the FA
path is NOT coopmat2. Whether gfx1151 (Radeon 8060S, RADV/Mesa 26.0.8) takes the
coopmat2 FA path is therefore the whole question, and it is not readable read-only.

This probe is the discriminator. Same request shapes as §4.1 / probes/
probe_cold_and_decode.py (streamed greedy 200-token decode, fresh salted prefix per
rung so every point is a cold prefill + decode). Expected outcomes:

  * 40k/80k materially above the §4.1 values, 13k flat  -> sparse FA is active on
    this box (non-cm2 path); #29639 is already paying for us.
  * the same ~15% decay as §4.1                        -> sparse FA is NOT active
    (cm2 path blocks quantized-KV sparse); then the lever is to force the
    non-coopmat2 FA path (operator env change), which this probe cannot test.

Read-only: HTTP GET/POST only, to the live server. Server pps is the rate source
(the router returns no usage token counts on streamed responses).
"""
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

# Same word pool as probes/probe_cold_and_decode.py so prompt text distribution is
# identical across probe generations (only the salt differs).
WORDS = ("the quick brown fox jumps over lazy dogs while data flows through "
         "distributed systems and kernels schedule workloads across "
         "heterogeneous accelerators in modern clusters").split()

# §4.1 rungs, so the comparison is point-for-point.
DECODE_CTX = [0, 13000, 40000, 80000]

# §4.1 reference values (Measured, b11391/b11399) for in-script deltas.
REF_4_1 = {0: 28.6, 13000: 27.4, 40000: 26.1, 80000: 24.3}


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


def chat_stream(prompt, max_tokens, timeout):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", payload, timeout)
    ttft = None
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
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.perf_counter() - t0
            last = obj
    return time.perf_counter() - t0, last, ttft


def main():
    salt = int(time.time())
    out = {"url": URL, "model": MODEL, "salt": salt,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "purpose": "decode vs ctx under b11430; sparse-FA (#29639) discriminator",
           "ref_4_1_b11391": REF_4_1,
           "decode_vs_ctx": []}
    print(f"probe_decode_vs_ctx_b11430  salt={salt}  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for target in DECODE_CTX:
        if target == 0:
            prompt = DECODE_CMD
        else:
            n_paras = max(1, round(target / TOKENS_PER_PARA))
            prompt = (make_prefix(salt + 9000 + target, n_paras)
                      + "\n" + DECODE_CMD)
        wall, obj, ttft = chat_stream(prompt, DECODE_TOKENS, timeout=1800)
        timings = (obj or {}).get("timings") or {}
        pred_pps = timings.get("predicted_per_second")
        decode_wall = wall - (ttft or 0.0)
        rate = DECODE_TOKENS / decode_wall if decode_wall else 0.0
        ref = REF_4_1.get(target)
        row = {"target": target,
               "prompt_tokens": ((obj or {}).get("usage") or {}).get("prompt_tokens", 0),
               "ttft_s": round(ttft or 0.0, 3),
               "decode_wall_s": round(decode_wall, 3),
               "decode_tok_s_wall": round(rate, 2),
               "server_pps": pred_pps,
               "ref_4_1_server_pps": ref,
               "delta_vs_4_1_pct": (round(100.0 * (pred_pps / ref - 1.0), 1)
                                    if (ref and pred_pps) else None)}
        out["decode_vs_ctx"].append(row)
        print(f"  ctx~{target:>6}  TTFT {(ttft or 0):7.2f}s  "
              f"decode {rate:5.2f} tok/s wall / {pred_pps} server pps  "
              f"(§4.1 ref {ref}, delta {row['delta_vs_4_1_pct']}%)", flush=True)
        time.sleep(2)

    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
