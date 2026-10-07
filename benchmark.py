#!/usr/bin/env python3
"""Benchmark + probe suite for the Strix Halo / llama.cpp setup described in
strix-halo-qwen3.8-perf.md.

Single stdlib-only entry point (Python 3.8+) for every measurement this repo
takes against the llama.cpp router at http://ms-s1-max-01:9931 running model
qwen3.8-flash-next. The probe scripts that previously lived under probes/
were merged in on 2026-10-06; the raw outputs they recorded stay under
probes/ and results/.

Modes (run `benchmark.py <mode> --help` for mode-specific flags; `bench` is
the default when the first argument is a flag, so `./benchmark.py
--skip-prefill` still works exactly as before):

  bench        Full benchmark:  A. introspection  B. idle TTFT  C. decode
               throughput  D. cold-prefill ladder  E. prefix-cache reuse
  cold         Genuinely-cold prefill ladder (clock-salted seeds) + decode
               rate vs context length   (was probes/probe_cold_and_decode.py)
  sparse-fa    Canary-gated decode-vs-depth sweep, the ggml-org/llama.cpp
               #29639 sparse-FA discriminator
               (was probes/probe_decode_sparse_fa.py, which superseded
               probes/probe_decode_vs_ctx_b11430.py)
  concurrency  Does aggregate decode throughput rise with concurrent slots?
               (was probes/probe_concurrency_scaling.py; perf doc §9 Q1)
  quiet        Contention gate: block until the box decodes the standard
               prompt at the standing rate --need times in a row. Run it
               before any long measurement
               (was probes/probe_wait_quiet.py)
  gguf-peek    Parse a GGUF header (metadata KV + tensor directory) from a
               local file or, via HTTP Range, a URL
               (was probes/gguf_peek_local.py + probes/gguf_header_peek.py)

Design notes (apply to all modes):
  * Ladder rungs use a distinct random seed each, so rungs are genuinely
    cold (no accidental prefix-cache hits from a shared word stream);
    bench's --salt and cold's clock-salting shift seeds between runs.
  * Token counts reported are those returned by the server in
    usage.prompt_tokens / timings, never assumed.
  * Decode rate is tokens / (wall_time - TTFT), which removes the fixed
    idle latency instead of guessing an allowance.
  * The box serves a production Hermes client: a uniform ~19-20 tok/s
    short-context decode means a co-tenant, not a regression. sparse-fa
    and concurrency bracket every sample with a canary decode; quiet is
    the gate. See perf doc §4.4.
"""
import argparse
import json
import os
import random
import statistics
import struct
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://ms-s1-max-01:9931"
DEFAULT_MODEL = "qwen3.8-flash-next"
DEFAULT_LADDER = [12000, 49000, 122000]
DEFAULT_DECODE_TOKENS = 200
DEFAULT_PREFIX_PARAS = 100  # ~13.4k tokens, matches probe.py's make_prefix(7)
DECODE_PROMPT = "Count from 1 to 80 separated by spaces."

# Canary floor: judged against the doc's standing short-context decode
# numbers (M1 26, §4.1 28.6, §4.2 28.55, results/4.json 28.4). A short-context
# decode below this means a co-tenant is contending for the decode pass and
# every number taken in that window is uninterpretable (perf doc §4.4).
CANARY_FLOOR = 26.0

# Measured tokens per generated paragraph (120 words + index) across several
# seeds: 134-138. Used only to size the ladder's text so the actual prompt
# lands near the target; the authoritative count is always usage.prompt_tokens.
TOKENS_PER_PARA = 135

# Fixed word pool so generated text (and therefore tokenisation) is reproducible
# and comparable to the recorded probe numbers. From probe.py.
WORDS = ("the quick brown fox jumps over lazy dogs while data flows through "
         "distributed systems and kernels schedule workloads across "
         "heterogeneous accelerators in modern clusters").split()

READY_PROMPT = "Reply with exactly the word READY.\n\nContext:\n"


def make_prefix(seed, n_paras, word_pool=WORDS):
    rng = random.Random(seed)
    lines = []
    for i in range(n_paras):
        line = " ".join(rng.choice(word_pool) for _ in range(120))
        lines.append(f"{line} {i}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTTP + SSE core (shared by every mode)
# --------------------------------------------------------------------------- #

def post(url, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def stream_chat(url, model, messages, max_tokens, timeout, *, temperature=0.0,
                enable_thinking=False, include_usage=False):
    """POST a streamed chat request and consume the SSE stream.

    Returns a dict with: wall_s, ttft_s, start_epoch/end_epoch (wall-clock,
    for cross-thread aggregate windows), last (final chunk carrying timings),
    usage, timings, finish_reason, content_pieces, content_chars.
    usage and timings are captured independently — llama.cpp delivers the
    usage chunk (only when include_usage is set) on a final chunk with no
    choices, and builds without stream_options support deliver timings
    without usage; never assume one object carries both (that assumption
    zeroed the concurrency probe's token counts once; perf doc §4.4)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    start_epoch = time.time()
    resp = post(url + "/v1/chat/completions", payload, timeout)
    ttft = None
    last = None
    usage = None
    finish = None
    pieces = 0
    chars = 0
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
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("timings"):
                last = obj
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    pieces += 1
                    chars += len(content)
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    wall = time.perf_counter() - t0
    return {
        "wall_s": wall,
        "ttft_s": ttft,
        "start_epoch": start_epoch,
        "end_epoch": time.time(),
        "last": last or {},
        "usage": usage or (last or {}).get("usage") or {},
        "timings": (last or {}).get("timings") or {},
        "finish_reason": finish,
        "content_pieces": pieces,
        "content_chars": chars,
    }


def chat_once(url, model, messages, max_tokens, timeout, temperature=0.0,
              stream=False, enable_thinking=False):
    """Non-streamed or streamed chat. Returns the tuple the bench sections
    use: (wall_s, response_obj, ttft_s or None, streamed_content_chars)."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    t0 = time.perf_counter()
    response = post(url + "/v1/chat/completions", payload, timeout)
    if stream:
        ttft = None
        last = {}
        n_chars = 0
        buf = ""
        for raw in response:
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
                last = obj
                choices = obj.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_chars += len(content)
        return time.perf_counter() - t0, last or {}, ttft, n_chars
    raw = response.read()
    obj = json.loads(raw.decode("utf-8"))
    return time.perf_counter() - t0, obj, None, 0


def timings_of(obj):
    t = obj.get("timings") or {}
    prompt_ms = t.get("prompt_ms", t.get("promptEvalDuration"))
    prompt_pps = t.get("prompt_per_second", t.get("promptEvalPerSecond"))
    pred_ms = t.get("predicted_ms", t.get("predictedEvalDuration"))
    pred_pps = t.get("predicted_per_second", t.get("predictedEvalPerSecond"))
    return prompt_ms, prompt_pps, pred_ms, pred_pps


def fmt_tok_per_s(numerator, seconds):
    if seconds and seconds > 0:
        return f"{numerator / seconds:.0f} tok/s"
    return "n/a"


def next_result_path(directory="results"):
    """Return directory/N.json where N is the smallest unused integer >= 1."""
    os.makedirs(directory, exist_ok=True)
    used = set()
    for name in os.listdir(directory):
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            used.add(int(stem))
    n = 1
    while n in used:
        n += 1
    return os.path.join(directory, f"{n}.json")


def section(title):
    print(f"\n=== {title} ===", flush=True)


def decode_canary(url, model, timeout=120):
    """Short greedy decode of the standard prompt; returns (server pps, full
    result). The standing-rate probe for contention detection."""
    r = stream_chat(url, model, [{"role": "user", "content": DECODE_PROMPT}],
                    DEFAULT_DECODE_TOKENS, timeout)
    return r["timings"].get("predicted_per_second"), r


# --------------------------------------------------------------------------- #
# mode: bench  (sections A-E, the perf doc's reference methodology)
# --------------------------------------------------------------------------- #

def run_introspection(url, model, timeout, results):
    section("A. Server introspection")
    health = get_json(url + "/health", timeout)
    results["health"] = health
    print(f"  health            : {health}")

    props = get_json(url + "/props", timeout)
    results["props"] = {k: props.get(k) for k in props
                        if k in ("build_info", "max_instances",
                                 "models_autoload", "default_generation_settings")}
    build = str(props.get("build_info") or "n/a")
    print(f"  build_info        : {build}")
    print(f"  max_instances     : {props.get('max_instances')}")
    print(f"  models_autoload   : {props.get('models_autoload')}")

    models = get_json(url + "/v1/models", timeout)
    target = next((m for m in models.get("data", []) if m["id"] == model), None)
    if target is None:
        print(f"  model '{model}'    : NOT FOUND in /v1/models")
        results["model_status"] = None
        return
    status = target.get("status", {})
    args = status.get("args", [])
    results["model_status"] = {"value": status.get("value"), "args": args}
    print(f"  model '{model}'    : {status.get('value')}")
    KEYS = ("--alias", "--batch-size", "--ctx-size", "--cache-ram",
            "--cache-type-k", "--cache-type-v", "--swa-checkpoints",
            "--flash-attn", "--kv-unified", "--load-mode", "--n-gpu-layers",
            "--parallel", "--ubatch-size", "--model", "--mmproj",
            "--model-draft", "--spec-type", "--spec-draft-model")
    # Flags are single tokens; values are the token that follows.
    pairs = list(zip(args, args[1:] + [None]))
    shown = {}
    for a, nxt in pairs:
        if a in KEYS:
            shown[a] = nxt if (nxt and not nxt.startswith("--")) else "(flag)"
    for k in KEYS:
        if k in shown:
            print(f"    {k:18s} = {shown[k]}")


def run_idle_ttft(url, model, timeout, results):
    section("B. Idle TTFT (time to first token, tiny request)")
    wall, obj, ttft, _ = chat_once(
        url, model, [{"role": "user", "content": "ping"}], max_tokens=1,
        timeout=timeout, stream=True)
    if ttft is None:
        print("  FAILED to stream a first token")
        results["idle_ttft_s"] = None
        return
    results["idle_ttft_s"] = ttft
    print(f"  TTFT              : {ttft*1000:.0f} ms")


def run_decode(url, model, decode_tokens, timeout, results):
    section(f"C. Decode throughput ({decode_tokens} tokens, greedy)")
    wall, obj, ttft, n_streamed = chat_once(
        url, model, [{"role": "user", "content": DECODE_PROMPT}],
        max_tokens=decode_tokens, timeout=timeout, stream=True)
    usage = obj.get("usage") or {}
    comp_tokens = usage.get("completion_tokens") or n_streamed or 0
    _, _, _, pred_pps = timings_of(obj)
    decode_wall = wall - (ttft or 0.0)
    results["decode"] = {
        "wall_s": wall, "ttft_s": ttft, "completion_tokens": comp_tokens,
        "decode_wall_s": decode_wall,
        "server_pps": pred_pps,
    }
    print(f"  TTFT              : {ttft*1000:.0f} ms" if ttft is not None
          else "  TTFT              : n/a")
    print(f"  total time        : {wall:.2f} s")
    print(f"  tokens            : {comp_tokens}")
    wall_rate = fmt_tok_per_s(comp_tokens, decode_wall)
    print(f"  decode rate       : {wall_rate} (wall: tokens/(total-TTFT))")
    if pred_pps:
        print(f"  decode rate       : {pred_pps:.0f} tok/s (server-reported)")


def run_prefill_ladder(url, model, ladder, timeout, results, salt=0):
    section("D. Cold prefill ladder")
    results["prefill_ladder"] = []
    for rung in ladder:
        n_paras = max(1, round(rung / TOKENS_PER_PARA))
        # Fixed seed per rung keeps rung text reproducible; --salt shifts it
        # so a rerun is genuinely cold instead of a prefix-cache hit (§4.1).
        seed = 4000 + rung + salt
        text = make_prefix(seed, n_paras)
        wall, obj, _, _ = chat_once(
            url, model, [{"role": "user", "content": READY_PROMPT + text}],
            max_tokens=4, timeout=timeout)
        usage = obj.get("usage") or {}
        pt = usage.get("prompt_tokens", 0)
        prompt_ms, prompt_pps, _, _ = timings_of(obj)
        row = {"target": rung, "prompt_tokens": pt, "wall_s": wall,
               "server_prompt_ms": prompt_ms, "server_pps": prompt_pps}
        results["prefill_ladder"].append(row)
        wall_rate = fmt_tok_per_s(pt, wall)
        line = f"  {rung//1000:>3}k target -> {pt:>6} prompt tokens, {wall:7.1f}s -> {wall_rate}"
        if prompt_pps:
            line += f" (server {prompt_pps:.0f} tok/s)"
        print(line, flush=True)


def run_cache_reuse(url, model, prefix_paras, timeout, results):
    section("E. Prefix-cache reuse")
    P = make_prefix(7, prefix_paras)        # fixed prefix, seed 7
    Q = make_prefix(99, prefix_paras)       # control prefix, never seen
    results["cache_reuse"] = []

    def one(label, text):
        wall, obj, _, _ = chat_once(
            url, model, [{"role": "user", "content": READY_PROMPT + text}],
            max_tokens=4, timeout=timeout)
        pt = (obj.get("usage") or {}).get("prompt_tokens", 0)
        row = {"label": label, "prompt_tokens": pt, "wall_s": wall}
        results["cache_reuse"].append(row)
        print(f"  {label:22s} : {pt:>6} tok, {wall:7.2f}s -> {fmt_tok_per_s(pt, wall)}",
              flush=True)
        return obj

    one("cold first pass", P)
    one("same prefix + suffix A", P + "\nSUFFIX-A What color is the sky?")
    one("same prefix + suffix B", P + "\nSUFFIX-B another new question here")
    one("control new prefix", Q)


def mode_bench(args, results):
    if not args.skip_introspection:
        run_introspection(args.url, args.model, args.timeout, results)
    run_idle_ttft(args.url, args.model, args.timeout, results)
    if not args.skip_decode:
        run_decode(args.url, args.model, args.decode_tokens, args.timeout,
                   results)
    if not args.skip_prefill:
        run_prefill_ladder(args.url, args.model, args.ladder, args.timeout,
                           results, salt=args.salt)
    if not args.skip_cache:
        run_cache_reuse(args.url, args.model, args.prefix_paras, args.timeout,
                        results)


# --------------------------------------------------------------------------- #
# mode: cold  (was probes/probe_cold_and_decode.py)
# --------------------------------------------------------------------------- #

def mode_cold(args):
    """Genuinely-cold prefill ladder + decode rate vs context length.

    bench's ladder reuses fixed per-rung seeds (4000+rung), so a second run
    collides with text already in the prefix cache and the "cold" wall times
    come back as cache hits (perf doc §4.1). This mode salts every seed with
    the wall clock: each rung is new text, and each decode-vs-context point
    carries a fresh salted prefix too (cold prefill + decode in one request).

    With swa-checkpoints 8 the per-token attention span is bounded, so a flat
    decode-vs-ctx curve is the default expectation; a slope shows KV/QSA-indexer
    reads competing with weight reads (§4.1 measured ~15% decay 8->80k ctx).
    Rate of record is the server's timings.predicted_per_second; usage counts
    are additionally requested (stream_options.include_usage) so completion
    token counts and a wall-clock rate are recorded alongside.
    """
    salt = int(time.time())
    out = {"mode": "cold", "probe": "benchmark.py cold",
           "url": args.url, "model": args.model,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "salt": salt, "cold_prefill": [], "decode_vs_ctx": []}
    print(f"cold  salt={salt}  {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for rung in args.ladder:
        n_paras = max(1, round(rung / TOKENS_PER_PARA))
        text = make_prefix(salt + rung, n_paras)
        wall, obj, _, _ = chat_once(
            args.url, args.model,
            [{"role": "user", "content": "Reply READY.\n\n" + text}],
            max_tokens=4, timeout=args.timeout)
        pt = (obj.get("usage") or {}).get("prompt_tokens", 0)
        _, p_ms, p_pps, _ = timings_of(obj)
        row = {"target": rung, "prompt_tokens": pt, "wall_s": round(wall, 3),
               "server_prompt_ms": p_ms, "server_pps": p_pps}
        out["cold_prefill"].append(row)
        print(f"  prefill {rung:>6} -> {pt:>6} tok  wall {wall:7.2f}s  "
              f"{pt/wall:.0f} tok/s (server {p_pps})", flush=True)

    for target in args.decode_ctx:
        if target == 0:
            prompt = DECODE_PROMPT
        else:
            n_paras = max(1, round(target / TOKENS_PER_PARA))
            prompt = (make_prefix(salt + 9000 + target, n_paras)
                      + "\n" + DECODE_PROMPT)
        r = stream_chat(args.url, args.model,
                        [{"role": "user", "content": prompt}],
                        args.decode_tokens, args.timeout, include_usage=True)
        pred_pps = r["timings"].get("predicted_per_second")
        comp = r["usage"].get("completion_tokens") or args.decode_tokens
        decode_wall = r["wall_s"] - (r["ttft_s"] or 0.0)
        row = {"target": target,
               "prompt_tokens": r["usage"].get("prompt_tokens", 0),
               "completion_tokens": comp,
               "ttft_s": round(r["ttft_s"] or 0.0, 3),
               "decode_wall_s": round(decode_wall, 3),
               "decode_tok_s": round(comp / decode_wall, 2) if decode_wall else None,
               "server_pps": pred_pps}
        out["decode_vs_ctx"].append(row)
        print(f"  decode ctx ~{target:>6} -> TTFT {row['ttft_s']:7.2f}s  "
              f"{row['decode_tok_s']} tok/s wall / {pred_pps} server pps",
              flush=True)
    return out


# --------------------------------------------------------------------------- #
# mode: sparse-fa  (was probes/probe_decode_sparse_fa.py, superseding
#                   probes/probe_decode_vs_ctx_b11430.py)
# --------------------------------------------------------------------------- #

def decode_once(url, model, prompt, timeout):
    r = stream_chat(url, model, [{"role": "user", "content": prompt}],
                    DEFAULT_DECODE_TOKENS, timeout)
    pps = r["timings"].get("predicted_per_second")
    gen = r["wall_s"] - (r["ttft_s"] or 0.0)
    return {"ttft_s": round(r["ttft_s"] or 0.0, 3),
            "server_pps": pps,
            "wall_pps": round(DEFAULT_DECODE_TOKENS / gen, 2) if gen > 0 else None,
            "at": time.strftime("%H:%M:%S")}


def mode_sparse_fa(args):
    """Decode rate vs context depth, canary-bracketed: the #29639 discriminator.

    Between b11399 and b11430 upstream merged ggml-org/llama.cpp#29639
    ("vulkan: sparse flash attention for quantized K/V"). Reported-by-source
    (RDNA3/RDNA4 discrete, NOT our box): with q8_0 K/V, Flash-Next previously
    ran DENSE attention over the whole context because Vulkan sparse FA only
    activated for f16 K/V; the PR opens the gate for quantized K/V at a
    separate 16x context/kept threshold and reports decode +15.6% @64k,
    +4.1% @128k, ~0 below 32k. Our preset has -ctk/-ctv q8_0, so sparse FA
    should be eligible above the threshold (16 * 2048 kept = 32768 cells)
    PROVIDED the FA path is not coopmat2 — RADV exposes
    VK_NV_cooperative_matrix2 only when radv_cooperative_matrix2_nv is set
    (default false), so on this box it should take coopmat1 and be eligible.
    This sweep measures whether the decode-vs-depth curve actually changes
    shape at the threshold:

      * 40k/80k materially above §4.1's 26.1 / 24.3 tok/s, 13k flat -> sparse
        FA is active on this box; #29639 is already paying for us.
      * the same ~15% decay as §4.1 -> sparse FA is NOT active (coopmat2
        blocks quantized-KV sparse); the lever becomes forcing the
        non-coopmat2 FA path, which this probe cannot test.

    Contamination control: every depth is bracketed by short-context canary
    decodes; a canary below CANARY_FLOOR marks the sample CONTENDED. The
    2026-10-06 16:54-17:36 window produced a uniform ~19.5 tok/s at every
    depth behind a co-tenant (§4.4) — an unbracketed number on this shared
    box is not interpretable. Default depths sit below / exactly on / above
    the 32k threshold inside one run. Fixed (un-salted) prefix per depth =>
    pass 1 is cold prefill, passes 2..N are prefix-cache hits, which makes
    repeated decode sampling cheap; --salt forces fresh text (all passes pay
    cold prefill). 122k is omitted on purpose: its ~7 min cold prefill is the
    most intrusive thing we can do to a production box and 80k already
    samples well above the threshold.
    """
    out = {"mode": "sparse-fa", "url": args.url, "model": args.model,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "depths": args.depths, "passes": args.passes,
           "canary_floor": CANARY_FLOOR, "samples": [], "canaries": []}
    print(f"sparse-fa  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"depths={args.depths} passes={args.passes} salt={args.salt}",
          flush=True)

    for p in range(args.passes):
        for depth in args.depths:
            if depth == 0:
                prompt = DECODE_PROMPT
            else:
                seed = (int(time.time()) + depth) if args.salt else 777000 + depth
                n_paras = max(1, round(depth / TOKENS_PER_PARA))
                prompt = make_prefix(seed, n_paras) + "\n" + DECODE_PROMPT
            pre, _ = decode_canary(args.url, args.model)
            pre_ok = (pre or 0) >= CANARY_FLOOR
            s = decode_once(args.url, args.model, prompt, args.timeout)
            post_, _ = decode_canary(args.url, args.model)
            post_ok = (post_ or 0) >= CANARY_FLOOR
            clean = pre_ok and post_ok
            rec = {"pass": p, "depth_target": depth, "clean": clean,
                   "canary_before": pre, "sample": s, "canary_after": post_}
            out["samples"].append(rec)
            out["canaries"].extend([{"server_pps": pre}, {"server_pps": post_}])
            print(f"  p{p} ctx~{depth:>6}  TTFT {s['ttft_s']:7.2f}s  "
                  f"decode {s['server_pps']} tok/s   "
                  f"canary {pre}/{post_} {'CLEAN' if clean else 'CONTENDED'}",
                  flush=True)
            json.dump(out, open(args.json, "w"), indent=2)

    print("\n=== summary (clean samples only) ===", flush=True)
    summary = {}
    for depth in args.depths:
        clean = [x["sample"]["server_pps"] for x in out["samples"]
                 if x["depth_target"] == depth and x["clean"]
                 and x["sample"]["server_pps"]]
        allp = [x["sample"]["server_pps"] for x in out["samples"]
                if x["depth_target"] == depth and x["sample"]["server_pps"]]
        summary[str(depth)] = {
            "n_clean": len(clean),
            "median_clean": statistics.median(clean) if clean else None,
            "n_all": len(allp),
            "median_all": statistics.median(allp) if allp else None}
        print(f"  ctx~{depth:>6}: clean n={summary[str(depth)]['n_clean']} "
              f"median {summary[str(depth)]['median_clean']}   "
              f"all n={summary[str(depth)]['n_all']} "
              f"median {summary[str(depth)]['median_all']}", flush=True)
    can = [c["server_pps"] for c in out["canaries"] if c["server_pps"]]
    summary["_canary"] = {"n": len(can),
                          "median": statistics.median(can) if can else None,
                          "min": min(can) if can else None}
    out["summary"] = summary
    print(f"  canary: n={len(can)} median {summary['_canary']['median']} "
          f"min {summary['_canary']['min']}", flush=True)
    json.dump(out, open(args.json, "w"), indent=2)
    print(f"\nResults written to {args.json}", flush=True)
    return None


# --------------------------------------------------------------------------- #
# mode: concurrency  (was probes/probe_concurrency_scaling.py)
# --------------------------------------------------------------------------- #

CONC_PROMPTS = [
    "Count from 1 to 80 separated by spaces.",
    "Count from 101 to 180 separated by spaces.",
    "Count from 201 to 280 separated by spaces.",
    "Count from 301 to 380 separated by spaces.",
    "Count from 401 to 480 separated by spaces.",
    "Count from 501 to 580 separated by spaces.",
]


def mode_concurrency(args):
    """Does aggregate decode throughput rise with concurrent slots?

    Client-side answer to perf doc §9 Q1 (bandwidth vs dequant/ALU ceiling)
    that needs no server-side profile. Mechanism: llama-server with
    `--parallel 2` does not decode each slot in its own forward pass — the
    scheduler appends the ready slots' tokens into ONE llama_batch, so a
    single trunk forward pass produces one token for every running slot.
    Weight reads (the dominant per-token cost of a 177B MoE at 4.25 bpw) are
    read once per pass and amortised across concurrent slots. Concurrency is
    therefore a knob on "memory traffic per output token" needing no server
    change, and its effect discriminates the two competing hypotheses:

      * bandwidth-bound (fixed bytes/s out of LPDDR5X):
            aggregate tok/s stays FLAT as slots are added; per-slot rate
            falls ~1/k.
      * per-pass-latency-bound (fixed time per forward pass: dispatch/fence
        latency, dequant ALU, MoE routing, QSA indexer scan, insufficient
        queue depth):
            aggregate tok/s RISES roughly k-fold while per-slot rate stays
            ~flat, until real bandwidth or compute saturation bends it over.

    It is also the directly actionable number for the production Hermes
    deployment: if aggregate scales with slots, `--parallel` is a throughput
    lever and the multi-slot cache-retention finding (Recipe C in
    notes/decode-mtp-operator-recipe.md) costs much less than it looks.

    Method: k concurrent streamed greedy 200-token decodes on short distinct
    prompts (each lands on its own slot; nothing else varies). Per-stream
    rate = completion_tokens / (wall - TTFT). Aggregate is reported two ways
    so the definitions are not confused: over the full wall window (includes
    each stream's own prefill = real goodput) and over the decode-only
    window. A lone-stream canary brackets every k; a depressed canary marks
    the point CONTENDED (§4.4). k stays <= 4 because the preset has
    --parallel 2; k above the slot count tests queueing, not batching, and
    adds load to a production box for no inference.
    """
    def one_stream(i):
        r = stream_chat(args.url, args.model,
                        [{"role": "user",
                          "content": CONC_PROMPTS[i % len(CONC_PROMPTS)]}],
                        args.max_tokens, args.timeout, include_usage=True)
        comp = r["usage"].get("completion_tokens") or 0
        srv_pps = r["timings"].get("predicted_per_second")
        srv_ms = r["timings"].get("predicted_ms")
        # token count, best to worst: explicit usage, then server-timings
        # derived, then the truncation guarantee (finish_reason "length"
        # means max_tokens was hit)
        if not comp and srv_pps and srv_ms:
            comp = int(round(srv_pps * srv_ms / 1000.0))
        if not comp and r["finish_reason"] == "length":
            comp = args.max_tokens
        gen = r["wall_s"] - (r["ttft_s"] or 0.0)
        return {"start": r["start_epoch"], "end": r["end_epoch"],
                "ttft_s": round(r["ttft_s"] or 0.0, 3),
                "completion_tokens": comp,
                "finish_reason": r["finish_reason"],
                "content_pieces": r["content_pieces"],
                "decode_s": round(gen, 3),
                "tok_s": round(comp / gen, 2) if gen > 0 else None,
                "server_pps": srv_pps}

    def run_k(k):
        res = [None] * k

        def work(i):
            try:
                res[i] = one_stream(i)
            except Exception as e:  # noqa: BLE001 - report, don't kill sweep
                res[i] = {"error": f"{type(e).__name__}: {e}"}
        th = [threading.Thread(target=work, args=(i,)) for i in range(k)]
        for t in th:
            t.start()
        for t in th:
            t.join()
        ok = [r for r in res if r and "error" not in r]
        if len(ok) != k:
            return {"k": k, "error": f"{k - len(ok)} of {k} streams failed",
                    "detail": res}
        total_tok = sum(r["completion_tokens"] for r in ok)
        wall = max(r["end"] for r in ok) - min(r["start"] for r in ok)
        dwin = (max(r["end"] for r in ok)
                - min(r["start"] + r["ttft_s"] for r in ok))
        return {"k": k,
                "per_stream_tok_s": [r["tok_s"] for r in ok],
                "per_stream_server_pps": [r["server_pps"] for r in ok],
                "ttft_s": [r["ttft_s"] for r in ok],
                "completion_tokens": [r["completion_tokens"] for r in ok],
                "total_tokens": total_tok,
                "aggregate_tok_s_wall": round(total_tok / wall, 2) if wall else None,
                "aggregate_tok_s_decode_window": round(total_tok / dwin, 2) if dwin else None,
                "median_stream_tok_s": statistics.median([r["tok_s"] for r in ok])}

    out = {"mode": "concurrency", "url": args.url, "model": args.model,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ks": args.ks, "canary_floor": CANARY_FLOOR, "samples": []}
    print(f"concurrency  {time.strftime('%Y-%m-%d %H:%M:%S')}  ks={args.ks}",
          flush=True)

    for p in range(args.passes):
        for k in args.ks:
            pre, _ = decode_canary(args.url, args.model, args.timeout)
            s = run_k(k)
            post_, _ = decode_canary(args.url, args.model, args.timeout)
            clean = ((pre or 0) >= CANARY_FLOOR
                     and (post_ or 0) >= CANARY_FLOOR and "error" not in s)
            rec = {"pass": p, "k": k, "clean": clean,
                   "canary_before": pre, "canary_after": post_, "sample": s}
            out["samples"].append(rec)
            agg = s.get("aggregate_tok_s_wall")
            med = s.get("median_stream_tok_s")
            print(f"  p{p} k={k}: per-stream median {med} tok/s  "
                  f"aggregate {agg} tok/s  "
                  f"canary {pre}/{post_} {'CLEAN' if clean else 'CONTENDED'}",
                  flush=True)
            json.dump(out, open(args.json, "w"), indent=2)

    print("\n=== summary: does aggregate scale with slots? (clean only) ===",
          flush=True)
    summ = {}
    base = None
    for k in args.ks:
        agg = [x["sample"]["aggregate_tok_s_wall"] for x in out["samples"]
               if x["k"] == k and x["clean"]
               and x["sample"].get("aggregate_tok_s_wall")]
        per = [x["sample"]["median_stream_tok_s"] for x in out["samples"]
               if x["k"] == k and x["clean"]
               and x["sample"].get("median_stream_tok_s")]
        ma = statistics.median(agg) if agg else None
        mp = statistics.median(per) if per else None
        if k == 1 and ma:
            base = ma
        summ[k] = {"n": len(agg), "aggregate": ma, "per_stream": mp,
                   "aggregate_scaling": round(ma / base, 2) if (ma and base) else None,
                   "per_stream_retention": round(mp / base, 2) if (mp and base) else None}
        print(f"  k={k}: clean n={summ[k]['n']}  aggregate {ma} tok/s "
              f"(x{summ[k]['aggregate_scaling']}, flat=1.00, linear={k}.00)  "
              f"per-stream {mp} tok/s (retention "
              f"{summ[k]['per_stream_retention']}, latency-bound=1.00, "
              f"bw-bound={round(1/k, 2)})", flush=True)
    out["summary"] = summ
    json.dump(out, open(args.json, "w"), indent=2)
    print(f"\nResults written to {args.json}", flush=True)
    return None


# --------------------------------------------------------------------------- #
# mode: quiet  (was probes/probe_wait_quiet.py)
# --------------------------------------------------------------------------- #

def mode_quiet(args):
    """Wait until ms-s1-max-01 looks idle, then say so. Bounded, read-only.

    Why: the box serves a production Hermes client. On 2026-10-06 an
    unguarded probe run returned a uniform ~19.5 tok/s at every context depth
    while a control decode minutes earlier returned 29 — a co-tenant, not a
    regression. Any absolute decode number taken during such a window is
    uninterpretable, so long probes should be gated on a quiet box instead of
    run blind and thrown away. A quiet box decodes the short greedy prompt at
    the standing rate the doc records (M1 26, §4.1 28.6, §4.2 28.55,
    results/4.json 28.4). Requires --need consecutive samples at or above
    --floor before declaring quiet.
    Exit 0 = quiet (safe to measure), 1 = still busy after --max-wait.
    """
    t_end = time.time() + args.max_wait
    streak = 0
    n = 0
    print(f"quiet  floor={args.floor} need={args.need} "
          f"max_wait={args.max_wait}s interval={args.interval}s", flush=True)
    while time.time() < t_end:
        pps = None
        wall = None
        try:
            pps, probe = decode_canary(args.url, args.model)
            wall = probe["wall_s"]
        except Exception as e:  # noqa: BLE001
            print(f"  {time.strftime('%H:%M:%S')} probe failed: "
                  f"{type(e).__name__}: {e}", flush=True)
        n += 1
        if pps and pps >= args.floor:
            streak += 1
        else:
            streak = 0
        print(f"  {time.strftime('%H:%M:%S')} sample {n}: "
              f"{pps} tok/s (wall {wall if wall is None else round(wall, 1)}s)  "
              f"streak {streak}/{args.need}", flush=True)
        if streak >= args.need:
            print(f"QUIET after {n} samples — safe to measure", flush=True)
            return 0
        time.sleep(args.interval)
    print(f"BUSY after {n} samples / {args.max_wait}s — do not measure",
          flush=True)
    return 1


# --------------------------------------------------------------------------- #
# mode: gguf-peek  (was probes/gguf_peek_local.py + gguf_header_peek.py)
# --------------------------------------------------------------------------- #
# Why: the #29811 MTP startup assert is decided by hparams.indexer_kpool,
# which qwen4exp load_hparams derives from the GGUF attention.compress_ratios
# array (src/models/qwen4exp.cpp:62-74 at b11430); graph_mtp then builds a
# k-pool input whenever indexer_kpool > 0 (qwen4exp.cpp:571) — including for
# the dense-attention draft block, which is the bug. If a draft GGUF's own
# metadata carries a compress-ratio array, editing it (gguf-set-metadata) is
# a potential no-rebuild lever. Only the header is needed: a URL read costs a
# few hundred KB via HTTP Range instead of the whole 4 GB file.
# --------------------------------------------------------------------------- #

GGUF_MAGIC = b"GGUF"
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
GGUF_SCALAR = {T_UINT8: (1, "<B"), T_INT8: (1, "<b"), T_UINT16: (2, "<H"),
               T_INT16: (2, "<h"), T_UINT32: (4, "<I"), T_INT32: (4, "<i"),
               T_FLOAT32: (4, "<f"), T_UINT64: (8, "<Q"), T_INT64: (8, "<q"),
               T_FLOAT64: (8, "<d")}


class _BytesReader:
    """Sequential reader over an in-memory buffer (local file prefix)."""

    def __init__(self, b):
        self.buf = b
        self.pos = 0

    def read(self, n):
        if self.pos + n > len(self.buf):
            raise EOFError(f"need {n} at {self.pos}, have {len(self.buf) - self.pos}")
        b = self.buf[self.pos:self.pos + n]
        self.pos += n
        return b

    def unpack(self, fmt):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


class _RangeReader:
    """Sequential reader over a URL, refilled by ranged HTTP GETs."""

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
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


def _gguf_string(r):
    ln = r.unpack("<Q")
    return r.read(ln).decode("utf-8", "replace")


def _gguf_val(r, vtype):
    if vtype in GGUF_SCALAR:
        return r.unpack(GGUF_SCALAR[vtype][1])
    if vtype == T_BOOL:
        return bool(r.unpack("<B"))
    if vtype == T_STRING:
        return _gguf_string(r)
    if vtype == T_ARRAY:
        elem = r.unpack("<I")
        cnt = r.unpack("<Q")
        return [_gguf_val(r, elem) for _ in range(cnt)]
    raise ValueError(f"unknown GGUF type {vtype} at offset {r.pos}")


def mode_gguf_peek(args):
    """Parse a GGUF header (metadata KV + tensor directory) from a local file
    or an http(s) URL (HTTP Range — talks to the host serving the file, not
    the inference box). Filters metadata keys by case-insensitive substring;
    prints the tensor directory when fully readable. A truncated local
    prefix, like probes/gguf/mtp_q80_head.bin's 4 MB cut, reports where it
    stopped instead of crashing (the old separate script crashed on it)."""
    src = args.source
    raw = None
    if src.startswith("http://") or src.startswith("https://"):
        r = _RangeReader(src)
    else:
        with open(src, "rb") as f:
            raw = f.read()
        r = _BytesReader(raw)
    subs = [s.lower() for s in args.keys]
    if r.read(4) != GGUF_MAGIC:
        print("not a GGUF")
        return 1
    ver = r.unpack("<I")
    n_tensors = r.unpack("<Q")
    n_kv = r.unpack("<Q")
    size_note = f"local {len(raw)} bytes" if raw is not None else "remote range-read"
    print(f"GGUF v{ver}  n_tensors={n_tensors}  n_kv={n_kv}  source={size_note}")
    hits = {}
    truncated = False
    print("--- metadata KV ---")
    try:
        for _ in range(n_kv):
            k = _gguf_string(r)
            vt = r.unpack("<I")
            v = _gguf_val(r, vt)
            low = k.lower()
            if not subs or any(s in low for s in subs):
                vs = str(v)
                hits[k] = vs[:300] + " ..." if len(vs) > 300 else vs
    except EOFError:
        truncated = True
    for k in sorted(hits):
        print(f"  {k} = {hits[k]}")
    if not args.no_tensors and not truncated:
        print("--- tensor directory ---")
        try:
            for i in range(n_tensors):
                name = _gguf_string(r)
                nd = r.unpack("<I")
                dims = struct.unpack(f"<{nd}Q", r.read(8 * nd))
                dt = r.unpack("<I")
                r.unpack("<Q")  # offset, unused
                print(f"  [{i:2}] {name:55s} dt={dt:3d} dims={list(dims)}")
        except EOFError:
            truncated = True
    if truncated:
        print(f"  [truncated: ran out of data at byte {r.pos} "
              f"(header prefix cut short of the full tensor directory)]")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

MODES = ("bench", "cold", "sparse-fa", "concurrency", "quiet", "gguf-peek")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="benchmark.py",
        description="Benchmark + probe suite for the Strix Halo llama.cpp "
                    "setup (strix-halo-qwen3.8-perf.md). Default mode is "
                    "`bench`, so flags without a mode still work.",
        epilog="modes: " + ", ".join(MODES) +
               " — see benchmark.py --help for the mode list and module "
               "docstring for methodology.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", metavar="MODE")

    def common(sp):
        sp.add_argument("--url", default=DEFAULT_URL)
        sp.add_argument("--model", default=DEFAULT_MODEL)
        sp.add_argument("--timeout", type=int, default=1800,
                        help="per-request HTTP timeout, seconds (default %(default)s)")

    p = sub.add_parser("bench", help="full benchmark A-E (default mode)")
    common(p)
    p.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER,
                   metavar="N",
                   help="prefill prompt-token targets, approximate (default: %(default)s)")
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument("--prefix-paras", type=int, default=DEFAULT_PREFIX_PARAS)
    p.add_argument("--salt", type=int, default=0,
                   help="shift the per-rung ladder seeds so a rerun is "
                        "genuinely cold instead of a prefix-cache hit")
    p.add_argument("--json", metavar="PATH",
                   help="write results to this JSON file instead of the "
                        "default results/N.json")
    p.add_argument("--skip-introspection", action="store_true")
    p.add_argument("--skip-decode", action="store_true")
    p.add_argument("--skip-prefill", action="store_true")
    p.add_argument("--skip-cache", action="store_true")

    p = sub.add_parser("cold",
                       help="cold prefill ladder (clock-salted) + decode vs ctx")
    common(p)
    p.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER)
    p.add_argument("--decode-ctx", type=int, nargs="+",
                   default=[0, 13000, 40000, 80000])
    p.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    p.add_argument("--json", metavar="PATH", default=None,
                   help="output JSON path (default: next results/N.json)")

    p = sub.add_parser("sparse-fa",
                       help="canary-gated decode-vs-depth (#29639 discriminator)")
    common(p)
    p.add_argument("--depths", type=int, nargs="+",
                   default=[0, 16000, 32000, 64000, 80000])
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--salt", action="store_true",
                   help="use fresh text per depth (all passes pay cold prefill)")
    p.add_argument("--json", metavar="PATH",
                   default="probes/probe_decode_sparse_fa.json")

    p = sub.add_parser("concurrency",
                       help="aggregate decode vs concurrent slots (perf doc §9 Q1)")
    common(p)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--json", metavar="PATH",
                   default="probes/probe_concurrency_scaling.json")

    p = sub.add_parser("quiet",
                       help="contention gate: block until the box is idle")
    common(p)
    p.add_argument("--floor", type=float, default=CANARY_FLOOR)
    p.add_argument("--need", type=int, default=2)
    p.add_argument("--max-wait", type=int, default=1800)
    p.add_argument("--interval", type=int, default=45)

    p = sub.add_parser("gguf-peek",
                       help="parse GGUF header from a local file or URL")
    p.add_argument("source", help="local path or http(s) URL")
    p.add_argument("keys", nargs="*", help="metadata key substrings to filter")
    p.add_argument("--no-tensors", action="store_true",
                   help="skip the tensor directory listing")

    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        argv = ["bench"]
    elif argv[0] not in MODES:
        if argv[0] in ("-h", "--help"):
            build_parser().print_help()
            return 0
        argv.insert(0, "bench")  # `./benchmark.py --skip-prefill` still works
    ap = build_parser()
    args = ap.parse_args(argv)
    results = None
    out_path = None

    if args.mode == "bench":
        results = {
            "url": args.url,
            "model": args.model,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "ladder_targets": args.ladder,
            "decode_tokens": args.decode_tokens,
        }
        print(f"Benchmark of {args.url} model={args.model} "
              f"({time.strftime('%Y-%m-%d %H:%M:%S %Z')})", flush=True)
        try:
            mode_bench(args, results)
        except Exception as e:  # noqa: BLE001 - top-level guard, save partial
            print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
            results["error"] = f"{type(e).__name__}: {e}"
            with open(args.json or next_result_path("results"), "w") as f:
                json.dump(results, f, indent=2)
            return 2
        out_path = args.json or next_result_path("results")
    elif args.mode == "quiet":
        return mode_quiet(args)
    elif args.mode == "gguf-peek":
        return mode_gguf_peek(args)
    elif args.mode == "cold":
        results = mode_cold(args)
        out_path = args.json or next_result_path("results")
    elif args.mode == "sparse-fa":
        return mode_sparse_fa(args)
    elif args.mode == "concurrency":
        return mode_concurrency(args)

    if results is not None and out_path:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}", flush=True)
    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
