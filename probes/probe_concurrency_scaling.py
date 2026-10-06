#!/usr/bin/env python3
"""Does aggregate decode throughput rise with concurrent slots?

This is the client-side answer to perf doc §9 Q1 ("what actually drives the
24-26 tok/s ceiling — memory bandwidth vs dequant/ALU cost"), which the doc records
as needing a server-side profile.  It does not need one.

Mechanism
---------
llama-server with `--parallel 2` (our preset) does not decode each slot in its own
forward pass: the scheduler appends the ready slots' tokens into ONE llama_batch, so
a single trunk forward pass produces one token for every running slot.  Weight reads
— the dominant per-token cost of a 177B MoE at 4.25 bpw — are therefore read ONCE per
pass and amortised across all concurrent slots.

So concurrency is a knob on "memory traffic per output token" that needs no server
change, and its effect discriminates the two competing hypotheses:

  * bandwidth-bound (fixed bytes/s out of LPDDR5X):
        aggregate tok/s stays FLAT as slots are added; per-slot rate falls ~1/k.
  * per-pass-latency-bound (fixed time per forward pass: dispatch/fence latency,
    dequant ALU, MoE routing, QSA indexer scan, insufficient queue depth):
        aggregate tok/s RISES roughly k-fold while per-slot rate stays ~flat, until
        real bandwidth or compute saturation bends it over.

Why the doc's two positions need this
--------------------------------------
§7 quotes drluoto attributing ~27 tok/s trunk decode to "the memory bandwidth of
this box".  notes/upstream-research §6 rebuts arithmetically — 6B active x 4.25 bpw
~ 3.2 GB/token, ~80 GB/s at 25 tok/s ~ 1/3 of the ~256 GB/s LPDDR5X-8000 peak — and
both sides are labelled Hypothesis.  The rebuttal has an unstated premise: that
gather/MoE access patterns achieve near-peak bandwidth.  This probe sidesteps the
bandwidth arithmetic entirely and measures the marginal value of an extra sequential
pass.

It is also the directly actionable number for the production Hermes deployment on
this box: if aggregate scales with slots, then `--parallel` is a throughput lever and
the multi-slot cache-retention finding (Recipe C) costs much less than it looks.

Method
------
- k concurrent streamed greedy 200-token decodes, short distinct prompts so each
  lands on its own slot and nothing else about the request varies.
- per-stream rate from the client: completion_tokens / (wall - TTFT).
- aggregate rate = total completion tokens / (last finish - first start), i.e. real
  goodput including each stream's own prefill, reported alongside a decode-window
  aggregate (total tokens / spread of decode-only windows) so the two definitions are
  not confused.
- a lone-stream canary brackets every k; a depressed canary marks the point
  CONTENDED.  On 2026-10-06 an ungated run produced a uniform ~19.5 tok/s at every
  depth because the box had a co-tenant — an unbracketed number on this server is
  not interpretable.
- k stays <= 4 because the preset has --parallel 2; k above the slot count tests
  queueing, not batching, and adds load to a production box for no inference.

Read-only: HTTP POST to the live server only.
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

URL = "http://ms-s1-max-01:9931"
MODEL = "qwen3.8-flash-next"
CANARY_FLOOR = 26.0
OUT_JSON = "probes/probe_concurrency_scaling.json"

PROMPTS = [
    "Count from 1 to 80 separated by spaces.",
    "Count from 101 to 180 separated by spaces.",
    "Count from 201 to 280 separated by spaces.",
    "Count from 301 to 380 separated by spaces.",
    "Count from 401 to 480 separated by spaces.",
    "Count from 501 to 580 separated by spaces.",
]


def chat_stream(prompt, max_tokens, timeout):
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    start = time.time()
    ttft = None
    last = None
    usage_seen = None
    finish = None
    pieces = 0
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
            usage_seen = obj.get("usage") or usage_seen  # may arrive without
            # timings on some builds; never rely on `last` holding both
            ch = obj.get("choices") or []
            if ch:
                if (ch[0].get("delta") or {}).get("content"):
                    pieces += 1
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                if ch[0].get("finish_reason"):
                    finish = ch[0]["finish_reason"]
            if obj.get("timings"):
                last = obj
    end = time.time()
    t = (last or {}).get("timings") or {}
    srv_pps = t.get("predicted_per_second")
    srv_ms = t.get("predicted_ms")
    # token count, best to worst: explicit usage, then server-timings derived,
    # then the truncation guarantee (finish_reason == "length" means max_tokens hit)
    comp = (usage_seen or {}).get("completion_tokens") or 0
    if not comp and srv_pps and srv_ms:
        comp = int(round(srv_pps * srv_ms / 1000.0))
    if not comp and finish == "length":
        comp = max_tokens
    gen = (time.perf_counter() - t0) - (ttft or 0.0)
    return {"start": start, "end": end,
            "ttft_s": round(ttft or 0.0, 3),
            "completion_tokens": comp,
            "finish_reason": finish,
            "content_pieces": pieces,
            "decode_s": round(gen, 3),
            "tok_s": round(comp / gen, 2) if gen > 0 else None,
            "server_pps": srv_pps}


def run_k(k, max_tokens, timeout):
    res = [None] * k
    def work(i):
        try:
            res[i] = chat_stream(PROMPTS[i % len(PROMPTS)], max_tokens, timeout)
        except Exception as e:  # noqa: BLE001 - report, do not kill the sweep
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
    dwin = max(r["end"] for r in ok) - min(r["start"] + r["ttft_s"] for r in ok)
    return {"k": k,
            "per_stream_tok_s": [r["tok_s"] for r in ok],
            "per_stream_server_pps": [r["server_pps"] for r in ok],
            "ttft_s": [r["ttft_s"] for r in ok],
            "completion_tokens": [r["completion_tokens"] for r in ok],
            "total_tokens": total_tok,
            "aggregate_tok_s_wall": round(total_tok / wall, 2) if wall else None,
            "aggregate_tok_s_decode_window": round(total_tok / dwin, 2) if dwin else None,
            "median_stream_tok_s": statistics.median([r["tok_s"] for r in ok])}


def canary(timeout):
    r = chat_stream("Count from 1 to 80 separated by spaces.", 200, timeout)
    return r.get("server_pps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    out = {"url": URL, "model": MODEL,
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "ks": args.ks, "canary_floor": CANARY_FLOOR, "samples": []}
    print(f"probe_concurrency_scaling  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"ks={args.ks}", flush=True)

    for p in range(args.passes):
        for k in args.ks:
            pre = canary(args.timeout)
            s = run_k(k, args.max_tokens, args.timeout)
            post = canary(args.timeout)
            clean = ((pre or 0) >= CANARY_FLOOR and (post or 0) >= CANARY_FLOOR
                     and "error" not in s)
            rec = {"pass": p, "k": k, "clean": clean,
                   "canary_before": pre, "canary_after": post, "sample": s}
            out["samples"].append(rec)
            agg = s.get("aggregate_tok_s_wall")
            med = s.get("median_stream_tok_s")
            print(f"  p{p} k={k}: per-stream median {med} tok/s  "
                  f"aggregate {agg} tok/s  (linear-from-1 would be "
                  f"{round((agg or 0),1) if k==1 else round((agg or 0),1)})  "
                  f"canary {pre}/{post} {'CLEAN' if clean else 'CONTENDED'}",
                  flush=True)
            json.dump(out, open(OUT_JSON, "w"), indent=2)

    print("\n=== summary: does aggregate scale with slots? (clean only) ===", flush=True)
    summ = {}
    base = None
    for k in args.ks:
        agg = [x["sample"]["aggregate_tok_s_wall"] for x in out["samples"]
               if x["k"] == k and x["clean"] and x["sample"].get("aggregate_tok_s_wall")]
        per = [x["sample"]["median_stream_tok_s"] for x in out["samples"]
               if x["k"] == k and x["clean"] and x["sample"].get("median_stream_tok_s")]
        ma = statistics.median(agg) if agg else None
        mp = statistics.median(per) if per else None
        if k == 1 and ma:
            base = ma
        summ[k] = {"n": len(agg), "aggregate": ma, "per_stream": mp,
                   "aggregate_scaling": round(ma / base, 2) if (ma and base) else None,
                   "per_stream_retention": round(mp / base, 2) if (mp and base) else None}
        print(f"  k={k}: clean n={summ[k]['n']}  aggregate {ma} tok/s "
              f"(x{summ[k]['aggregate_scaling']}, flat=1.00, linear={k}.00)  "
              f"per-stream {mp} tok/s (retention {summ[k]['per_stream_retention']}, "
              f"latency-bound=1.00, bw-bound={round(1/k,2)})", flush=True)
    out["summary"] = summ
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
