#!/usr/bin/env python3
"""Decode rate vs context depth on b11430, with a contention canary.

Two reasons this probe exists.

1. #29639 discriminator.  Between b11399 and b11430 upstream merged ggml-org/
   llama.cpp#29639 ("vulkan: sparse flash attention for quantized K/V", merged
   2026-10-05 08:37Z).  Reported-by-source (PR body, RDNA3/RDNA4 discrete cards,
   NOT our box): with a q8_0 KV cache Flash-Next ran DENSE attention over the whole
   context because Vulkan sparse FA only activated for f16 K/V; the PR opens the
   gate for quantized K/V with a separate 16x context/kept threshold, and reports
   decode +15.6% @64k, +4.1% @128k, ~0 below 32k.

   The gate at 8345f3339 (b11430) ggml-vulkan.cpp:8149 is:

       min_ratio   = path == FA_COOPMAT2 ? 4 : (kv_f16 ? 2 : 16)
       use_sparse  = ... (kv_f16 || path != FA_COOPMAT2) && nem0 == KV
                     && KV >= max(4096, min_ratio * n_kv_max) && ...

   Our preset has -ctk/-ctv q8_0, so kv_f16 is false and sparse FA needs a
   NON-coopmat2 FA path.  Path choice (ggml-vulkan.cpp:1340) is
   `device->coopmat2 ? FA_COOPMAT2 : coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR`,
   and RADV only exposes VK_NV_cooperative_matrix2 when `radv_cooperative_matrix2_nv`
   is set, default false (Mesa docs; RADV on gfx1151 reports `matrix cores:
   KHR_coopmat` i.e. coopmat1 in community --list-devices output).  So sparse FA
   SHOULD be eligible on this box above the 16x threshold.  This probe measures
   whether the decode-vs-depth curve actually changes shape at the threshold.

2. Contamination control.  probes/probe_decode_vs_ctx_b11430.py run 2026-10-06
   16:54 returned ~20 tok/s at EVERY depth (short included) while a control decode
   minutes later on the same build returned 29.  The box serves a production
   Hermes client, so a single-pass number is not trustworthy.  Every depth here is
   therefore bracketed by a short-context canary decode; if the canary is depressed
   the point is flagged, and each depth is sampled repeatedly (deep depths are cheap
   to re-sample because the prefix stays in the prompt cache after the first pass).

Method: fixed (un-salted) prefix per depth => pass 1 is cold prefill, passes 2..N
are prefix-cache hits, which makes repeated decode sampling cheap.  Rate source is
the server's timings.predicted_per_second (the router returns no usage counts on
streamed responses).  Greedy, 200 tokens, enable_thinking false — same shape as
benchmark.py §C / perf doc §4.1.

Read-only: HTTP GET/POST to the live server only.
"""
import argparse
import json
import random
import statistics
import sys
import time
import urllib.request

URL = "http://ms-s1-max-01:9931"
MODEL = "qwen3.8-flash-next"
TOKENS_PER_PARA = 135
DECODE_TOKENS = 200
DECODE_CMD = "Count from 1 to 80 separated by spaces."
# Canary is judged against the doc's standing short-context decode numbers
# (M1 26, §4.1 28.6, §4.2 28.55, results/4.json 28.4, re-check 2026-10-06 29.0).
CANARY_FLOOR = 26.0

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
            if choices and (choices[0].get("delta") or {}).get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
            last = obj
    return time.perf_counter() - t0, last, ttft


def decode_once(prompt, timeout):
    wall, obj, ttft = chat_stream(prompt, DECODE_TOKENS, timeout)
    t = (obj or {}).get("timings") or {}
    return {"ttft_s": round(ttft or 0.0, 3),
            "server_pps": t.get("predicted_per_second"),
            "wall_pps": round(DECODE_TOKENS / (wall - (ttft or 0.0)), 2)
            if wall > (ttft or 0.0) else None,
            "at": time.strftime("%H:%M:%S")}


def prompt_for(depth, salted):
    if depth == 0:
        return DECODE_CMD
    # fixed seed per depth => stable text across runs => cache hits on repeat
    seed = (int(time.time()) + depth) if salted else 777000 + depth
    n_paras = max(1, round(depth / TOKENS_PER_PARA))
    return make_prefix(seed, n_paras) + "\n" + DECODE_CMD


def main():
    ap = argparse.ArgumentParser()
    # 16k sits BELOW the 16x quantized-KV sparse threshold (16*2048 kept = 32768
    # cells), 32k sits exactly on it, 64k/80k above — so the threshold's effect on
    # the decode curve is visible inside one run.  122k is omitted on purpose: its
    # ~7 min cold prefill is the most intrusive thing we can do to a production box
    # and 80k already samples well above the threshold.
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[0, 16000, 32000, 64000, 80000])
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--salt", action="store_true",
                    help="use fresh text per depth (all passes pay cold prefill)")
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    out = {"url": URL, "model": MODEL,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "depths": args.depths, "passes": args.passes,
           "canary_floor": CANARY_FLOOR, "samples": [], "canaries": []}
    print(f"probe_decode_sparse_fa  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"depths={args.depths} passes={args.passes} salt={args.salt}", flush=True)

    for p in range(args.passes):
        for depth in args.depths:
            pre = decode_once(DECODE_CMD, args.timeout)
            pre_ok = (pre["server_pps"] or 0) >= CANARY_FLOOR
            s = decode_once(prompt_for(depth, args.salt), args.timeout)
            post_ = decode_once(DECODE_CMD, args.timeout)
            post_ok = (post_["server_pps"] or 0) >= CANARY_FLOOR
            clean = pre_ok and post_ok
            rec = {"pass": p, "depth_target": depth, "clean": clean,
                   "canary_before": pre, "sample": s, "canary_after": post_}
            out["samples"].append(rec)
            out["canaries"].extend([pre, post_])
            print(f"  p{p} ctx~{depth:>6}  TTFT {s['ttft_s']:7.2f}s  "
                  f"decode {s['server_pps']} tok/s   "
                  f"canary {pre['server_pps']}/{post_['server_pps']} "
                  f"{'CLEAN' if clean else 'CONTENDED'}", flush=True)
            json.dump(out, open(OUT_JSON, "w"), indent=2)

    print("\n=== summary (clean samples only) ===", flush=True)
    summary = {}
    for depth in args.depths:
        clean = [x["sample"]["server_pps"] for x in out["samples"]
                 if x["depth_target"] == depth and x["clean"]
                 and x["sample"]["server_pps"]]
        allp = [x["sample"]["server_pps"] for x in out["samples"]
                if x["depth_target"] == depth and x["sample"]["server_pps"]]
        summary[depth] = {"n_clean": len(clean),
                          "median_clean": statistics.median(clean) if clean else None,
                          "n_all": len(allp),
                          "median_all": statistics.median(allp) if allp else None}
        print(f"  ctx~{depth:>6}: clean n={summary[depth]['n_clean']} "
              f"median {summary[depth]['median_clean']}   "
              f"all n={summary[depth]['n_all']} median {summary[depth]['median_all']}",
              flush=True)
    can = [c["server_pps"] for c in out["canaries"] if c["server_pps"]]
    summary["_canary"] = {"n": len(can),
                          "median": statistics.median(can) if can else None,
                          "min": min(can) if can else None}
    out["summary"] = summary
    print(f"  canary: n={len(can)} median {summary['_canary']['median']} "
          f"min {summary['_canary']['min']}", flush=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    return 0


OUT_JSON = "probes/probe_decode_sparse_fa.json"

if __name__ == "__main__":
    sys.exit(main())
