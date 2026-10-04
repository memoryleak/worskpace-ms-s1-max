#!/usr/bin/env python3
"""Full benchmark of the Strix Halo / llama.cpp setup described in
strix-halo-qwen3.8-perf.md.

Targets the llama.cpp router at http://ms-s1-max-01:9931 running model
qwen3.8-flash-next. Uses only the Python stdlib so it runs anywhere with
Python 3.8+. Request shapes mirror the probe scripts in ~/strix-perf-probes/,
so wall-clock numbers are comparable with the values recorded in the perf doc.

What it measures (see the doc §4 for the reference methodology):

  A. Server introspection   /health, /props, /v1/models -> loaded state + key args
  B. Idle TTFT              time-to-first-token on a tiny streamed request
  C. Decode throughput      greedy 200-token generation, tok/s (wall + server-reported)
  D. Cold prefill ladder    12k / 49k / 122k token prompts, distinct text per rung,
                            tok/s (wall + server-reported)
  E. Prefix-cache reuse     fixed prefix: cold pass, cache-hit pass, control cold pass

Design notes:
  * Ladder rungs use a distinct random seed each, so later rungs are genuinely
    cold (no accidental prefix-cache hits from a shared word stream).
  * Token counts reported are those returned by the server in usage.prompt_tokens,
    never assumed.
  * Decode rate is tokens / (wall_time - TTFT), which removes the fixed idle
    latency instead of guessing an allowance.

Usage:
  benchmark.py [--url URL] [--model NAME] [--ladder N N N] [--decode-tokens N]
               [--prefix-paras N] [--timeout S] [--json PATH]
               [--skip-introspection] [--skip-decode] [--skip-prefill] [--skip-cache]
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://ms-s1-max-01:9931"
DEFAULT_MODEL = "qwen3.8-flash-next"
DEFAULT_LADDER = [12000, 49000, 122000]
DEFAULT_DECODE_TOKENS = 200
DEFAULT_PREFIX_PARAS = 100  # ~13.4k tokens, matches probe.py's make_prefix(7)

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
DECODE_PROMPT = "Count from 1 to 80 separated by spaces."


def make_prefix(seed, n_paras, word_pool=WORDS):
    rng = random.Random(seed)
    lines = []
    for i in range(n_paras):
        line = " ".join(rng.choice(word_pool) for _ in range(120))
        lines.append(f"{line} {i}")
    return "\n".join(lines)


def post(url, payload, timeout):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def _stream_decode(response, timeout):
    """Iterate a streamed SSE response, yielding decoded JSON objects.

    Returns (first_token_s, n_streamed_tokens, n_chunks) via a small state
    object; caller reads the yielded dicts for usage/timings."""
    first_token_s = None
    t0 = time.perf_counter()
    n_tokens = 0
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
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - t0
                    n_tokens += len(content)
            yield obj, first_token_s, n_tokens


def chat_once(url, model, messages, max_tokens, timeout, temperature=0.0,
              stream=False, enable_thinking=False):
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
        last = None
        ttft = None
        n_streamed = 0
        for obj, first_token_s, n_tokens in _stream_decode(response, timeout):
            last = obj
            if first_token_s is not None:
                ttft = first_token_s
            n_streamed = n_tokens
        wall = time.perf_counter() - t0
        return wall, last or {}, ttft, n_streamed
    raw = response.read()
    obj = json.loads(raw.decode("utf-8"))
    wall = time.perf_counter() - t0
    return wall, obj, None, 0


def next_result_path(directory="results"):
    """Return directory/N.json where N is the next free positive integer.

    Scans the directory for files whose stem is an integer and picks the
    smallest integer >= 1 that is not already used, creating ``directory``
    if it does not exist.
    """
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


def get_json(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


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


def section(title):
    print(f"\n=== {title} ===", flush=True)


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
            "--parallel", "--ubatch-size", "--model", "--mmproj", "--model-draft")
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


def run_prefill_ladder(url, model, ladder, timeout, results):
    section("D. Cold prefill ladder")
    results["prefill_ladder"] = []
    for rung in ladder:
        n_paras = max(1, round(rung / TOKENS_PER_PARA))
        seed = 4000 + rung
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


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Full benchmark of the Strix Halo llama.cpp setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--ladder", type=int, nargs="+", default=DEFAULT_LADDER,
                    metavar="N",
                    help="prefill prompt-token targets, approximate (default: %(default)s)")
    ap.add_argument("--decode-tokens", type=int, default=DEFAULT_DECODE_TOKENS)
    ap.add_argument("--prefix-paras", type=int, default=DEFAULT_PREFIX_PARAS)
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-request HTTP timeout, seconds (default %(default)s)")
    ap.add_argument("--json", metavar="PATH",
                    help="write results to this JSON file instead of the "
                         "default results/N.json")
    ap.add_argument("--skip-introspection", action="store_true")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--skip-cache", action="store_true")
    args = ap.parse_args(argv)

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
        if not args.skip_introspection:
            run_introspection(args.url, args.model, args.timeout, results)
        run_idle_ttft(args.url, args.model, args.timeout, results)
        if not args.skip_decode:
            run_decode(args.url, args.model, args.decode_tokens,
                       args.timeout, results)
        if not args.skip_prefill:
            run_prefill_ladder(args.url, args.model, args.ladder,
                               args.timeout, results)
        if not args.skip_cache:
            run_cache_reuse(args.url, args.model, args.prefix_paras,
                            args.timeout, results)
    except urllib.error.URLError as e:
        print(f"\nERROR: request failed: {e}", file=sys.stderr)
        results["error"] = str(e)
        return 2
    except Exception as e:  # noqa: BLE001 - top-level guard, report and exit
        print(f"\nERROR: {type(e).__name__}: {e}", file=sys.stderr)
        results["error"] = f"{type(e).__name__}: {e}"
        return 2

    out_path = args.json or next_result_path("results")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}", flush=True)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())