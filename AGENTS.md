# AGENTS.md

This repo is a performance investigation workspace for the local inference box
`ms-s1-max-01` (AMD Ryzen AI MAX+ 395 / 128 GiB, llama.cpp router at
`http://ms-s1-max-01:9931/v1`, Hermes provider `custom:ms-s1-max`, model
`qwen3.8-flash-next`).

## Required first step

Read and parse `strix-halo-qwen3.8-perf.md` in this directory before doing
anything else. It is the authoritative record of the current state and its
conventions are binding:

- **Measured** (measurement runs this session, via `benchmark.py`) vs **Reported-by-source** (quoted from
  an upstream thread/PR, not reproduced here) vs **Hypothesis** (causal claim
  not isolated) vs **Speculation/advice** (excluded from the doc). When you
  write anything new, keep that taxonomy. Do not silently upgrade a
  reported-by-source claim to a measured fact, or a hypothesis to a conclusion.
- Open questions are listed only in §9 as questions. Do not assert them as
  findings.

## Baseline facts you must work from (all from that file)

- Decode is stuck at 24–26 tok/s across every config state tested; batch and
  cache changes did not move it. §7 attributes ~27 tok/s trunk decode to memory
  bandwidth.
- Cold prefill degrades with prompt length as found (407 → 357 → 218 tok/s at
  12k/49k/122k), and roughly flattened after the "Fix #1" bundle (q8_0 KV,
  parallel 2, ubatch 2048): M6 448/459/373 tok/s.
- Prefix-cache hits are real and cheap (7.9–11.2 s for 54k/105k repeats).
- A spec-decoding branch (ggml-org/llama.cpp#28512) reports 33–58 tok/s decode
  on the same APU + a draft file `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` from
  huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF, untested on our stock
  build b11391.

## Task

Research and suggest improvements, then (if asked) apply and re-measure them.
Priority order:

1. **Decode throughput** — the dominant limiter. Investigate what actually
   drives the 24–26 tok/s ceiling (memory bandwidth vs dequant/ALU cost), and
   which concrete changes could raise it: MTP/speculative drafting, quant
   format, kernel changes from the #28512 branch lineage.
2. **Long-prompt prefill** — the 122k-token degradation mechanism (prompt-cache
   eviction vs other cause) was never isolated; the fixes were bundled.
3. **Open questions in §9 of the doc** — these are the unverified gaps. Work
   them in order and report each as measured / reported / hypothesis as
   appropriate.

## Workspace boundary (hard rule)

All work happens **inside `/home/hermes/Projects/strix-halo` only**:

- Every file you create or edit — notes, probe scripts, captured output,
  config dumps, results — goes in this directory. Do not write anywhere else on the machine.
- The inference box `ms-s1-max-01` is **read-only from here**: you may send
  HTTP requests to `http://ms-s1-max-01:9931/v1` to probe/measure, but never
  modify the server — no preset edits, no model-draft load, no container
  reload, no `podman exec`, no `ssh`. It serves a production Hermes client.
  Anything that would change server state goes in this directory as a
  *proposed* patch/recipe with a note that the operator must apply it.
- External lookups (web search, upstream threads/PRs, huggingface) are allowed
  for research; save what you find into this directory, never elsewhere.

## Rules
- Always state provenance: never claim something measured that you only read.
- Record results back into `strix-halo-qwen3.8-perf.md` under the same
  conventions; do not replace the document with a summary.