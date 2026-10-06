#!/usr/bin/env python3
"""Wait until ms-s1-max-01 looks idle, then say so. Bounded, read-only.

Why: the box serves a production Hermes client.  On 2026-10-06 an unguarded probe
run returned a uniform ~19.5 tok/s at every context depth while a control decode
minutes earlier returned 29 — a co-tenant, not a regression.  Any absolute decode
number taken during such a window is uninterpretable, so long probes should be
gated on a quiet box instead of run blind and thrown away.

A quiet box decodes the short greedy prompt at the standing rate the doc records
(M1 26, §4.1 28.6, §4.2 28.55, results/4.json 28.4).  We require `--need` consecutive
samples at or above `--floor` before declaring it quiet.

Usage: probe_wait_quiet.py [--floor 26] [--need 2] [--max-wait 1800] [--interval 45]
Exit 0 = quiet (safe to measure), 1 = still busy after max-wait (do not measure).
"""
import argparse
import json
import sys
import time
import urllib.request

URL = "http://ms-s1-max-01:9931"
MODEL = "qwen3.8-flash-next"


def decode_pps(timeout=120):
    payload = {"model": MODEL,
               "messages": [{"role": "user",
                             "content": "Count from 1 to 80 separated by spaces."}],
               "max_tokens": 200, "temperature": 0.0, "stream": True,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        URL + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=timeout)
    last = None
    buf = ""
    for raw in resp:
        buf += raw.decode("utf-8", "replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.strip()
            if not line.startswith("data:"):
                continue
            d = line[5:].strip()
            if d == "[DONE]":
                continue
            try:
                last = json.loads(d)
            except json.JSONDecodeError:
                continue
    t = (last or {}).get("timings") or {}
    return t.get("predicted_per_second"), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=26.0)
    ap.add_argument("--need", type=int, default=2)
    ap.add_argument("--max-wait", type=int, default=1800)
    ap.add_argument("--interval", type=int, default=45)
    args = ap.parse_args()

    t_end = time.time() + args.max_wait
    streak = 0
    n = 0
    while time.time() < t_end:
        pps = None
        wall = None
        try:
            pps, wall = decode_pps()
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
    print(f"BUSY after {n} samples / {args.max_wait}s — do not measure", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
