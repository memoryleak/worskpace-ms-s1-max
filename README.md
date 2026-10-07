# Strix Halo benchmark

`benchmark.py` — single stdlib-only entry point for every measurement of the
llama.cpp setup in `strix-halo-qwen3.8-perf.md` (model `qwen3.8-flash-next`
on `http://ms-s1-max-01:9931`). Python 3.8+, no dependencies.

The probe scripts that previously lived under `probes/` are merged in as
modes (2026-10-06). `probes/` and `results/` now hold only recorded raw
output. `bench` is the default mode, so bare-flag invocations keep working.

Modes (`./benchmark.py <mode> --help` for flags):

    bench        A. Server introspection   /health, /props, /v1/models
                   (loaded state + key args)
                 B. Idle TTFT              time to first token, tiny request
                 C. Decode throughput      greedy 200-token, tok/s (wall + server)
                 D. Cold prefill ladder    12k / 49k / 122k, distinct text per
                                           rung, tok/s (wall + server)
                 E. Prefix-cache reuse     cold pass, cache-hit pass, control
    cold         clock-salted (always-cold) prefill ladder + decode rate vs
                 context length            (was probes/probe_cold_and_decode.py)
    sparse-fa    canary-gated decode-vs-depth sweep — the #29639 sparse-FA
                 discriminator             (was probes/probe_decode_sparse_fa.py)
    concurrency  aggregate decode throughput vs concurrent slots — answers
                 perf doc §9 Q1            (was probes/probe_concurrency_scaling.py)
    quiet        contention gate: block until the box decodes the standard
                 prompt at ≥26 tok/s --need times in a row; exit 0 = idle
                                             (was probes/probe_wait_quiet.py)
    gguf-peek    parse a GGUF header (metadata + tensor directory) from a
                 local file or URL via HTTP Range
                                             (was probes/gguf_peek_local.py +
                                              probes/gguf_header_peek.py)

Examples:

    ./benchmark.py                          # full bench (ladder ~7-8 min)
    ./benchmark.py --skip-prefill --skip-cache   # quick smoke test
    ./benchmark.py cold                     # always-cold ladder + decode vs ctx
    ./benchmark.py quiet && ./benchmark.py sparse-fa   # gated probe run
    ./benchmark.py concurrency              # §9 Q1 slot-scaling sweep
    ./benchmark.py gguf-peek probes/gguf/mtp_q80_head.bin compress

Shared options: `--url`, `--model`, `--timeout S`, `--json PATH`. In `bench`:
`--ladder N N N`, `--decode-tokens N`, `--prefix-paras N`, `--salt N` (shift
ladder seeds; use a new value per run so rungs stay cold — an unsalted rerun
re-sends text already in the prefix cache and reports cache-hit walls), and
`--skip-introspection/--skip-decode/--skip-prefill/--skip-cache`.

Notes on methodology:
- Ladder rungs use a distinct random seed each, so rungs are genuinely cold
  rather than prefix-cache hits from a shared word stream; `--salt` (bench)
  and clock-salting (cold) shift those seeds between runs.
- The box serves a production Hermes client: gate long measurement runs on
  `./benchmark.py quiet` (exit 0 = idle) and treat a uniform ~19-20 tok/s
  short-context decode as contention, not a regression (perf doc §4.4).
  `sparse-fa` and `concurrency` bracket every sample with a canary decode
  and mark CONTENDED samples as unusable.
- Reported token counts are the server's `usage.prompt_tokens` / timings,
  never assumed. Streamed probes request `stream_options.include_usage`;
  usage and timings are captured independently because the server may
  deliver them on different chunks.
- Decode rate = tokens / (wall time - TTFT). Idle TTFT is measured directly
  rather than subtracting a fixed allowance (the doc used a 0.1 s allowance).
