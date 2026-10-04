# Strix Halo benchmark

`benchmark.py` — full benchmark of the llama.cpp setup in
`~/strix-halo-qwen3.8-perf.md` (model `qwen3.8-flash-next` on
`http://ms-s1-max-01:9931`). Stdlib-only Python, no dependencies.

It measures the same quantities the perf doc records:

  A. Server introspection   /health, /props, /v1/models (loaded state + key args)
  B. Idle TTFT              time to first token on a tiny streamed request
  C. Decode throughput      greedy 200-token generation, tok/s (wall + server)
  D. Cold prefill ladder    12k / 49k / 122k token prompts, cold (distinct text
                            per rung), tok/s (wall + server-reported)
  E. Prefix-cache reuse     fixed prefix: cold pass, cache-hit pass, control

Run the full benchmark (prefill ladder takes ~7-8 min at current speeds):

    ./benchmark.py

Quick smoke test (introspection + TTFT + decode only):

    ./benchmark.py --skip-prefill --skip-cache

Single prefill rung at ~12k tokens:

    ./benchmark.py --skip-decode --ladder 12000

Options: `--url`, `--model`, `--ladder N N N`, `--decode-tokens N`,
`--prefix-paras N`, `--timeout S`, `--json PATH` (also dump raw results),
and `--skip-introspection/--skip-decode/--skip-prefill/--skip-cache`.

Notes on methodology:
- Ladder rungs use a distinct random seed each, so later rungs are genuinely
  cold rather than prefix-cache hits from a shared word stream.
- Reported token counts are the server's `usage.prompt_tokens` / timings,
  never assumed.
- Decode rate = tokens / (wall time - TTFT). Idle TTFT is measured directly
  rather than subtracting a fixed allowance (the doc used a 0.1 s allowance).