# Strix Halo 128 GB — Qwen 3.8 Flash Next: Performance & Timeout Investigation

**Date:** 2026-10-04 (all times CEST) · Client: i5-9500T box running Hermes → GPU box `ms-s1-max-01` (Ryzen AI MAX+ 395, 128 GiB), llama.cpp router, port 9931. 

> **Conventions of this document.** Measured = observed directly by measurement runs logged this session. Reported-by-source = quoted from the cited upstream thread/PR, not reproduced by us. Hypothesis = causal explanation we did not isolate (fixes were applied as bundles). Speculation, predictions, and advice are excluded; open questions are listed only as questions.

## 1. Baseline hardware & software (from lshw, uname, /proc/cmdline, dmesg, vulkaninfo, --list-devices — all run 2026-10-04)

| Item | Value |
|---|---|
| Inference box | `ms-s1-max-01`, chassis "MS-S1 MAX" board MGSHWSA, AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |
| Memory | 128 GiB: 8 × 16 GiB "Synchronous Unbuffered (Unregistered) 8000 MHz" (lshw) |
| CPU caches | 1280 KiB L1, 16 MiB L2, 64 MiB L3 (lshw) |
| GPU | `amdgpu` DRM driver (dmesg), PCI bd:00.0, device 0x1002:0x1586, IP-discovery `soc21_common`/`gmc_v11_0`; integrated |
| Userspace GPU stack | Vulkan / Mesa **RADV** — `vulkaninfo`: `Radeon 8060S Graphics (RADV STRIX_HALO)`, `DRIVER_ID_MESA_RADV`, Mesa 26.0.8-1ubuntu0.3, apiVersion 1.4.335, conformance 1.4.0.0. (A second Vulkan device, llvmpipe CPU, exists; `llama-server --list-devices` lists only `Vulkan0`.) |
| llama.cpp view | `podman exec llama-server llama-server --list-devices` → `Vulkan0: AMD Radeon 8060S Graphics (RADV STRIX_HALO) (128000 MiB, 54127 MiB free)` |
| Kernel | Ubuntu, Linux 7.0.0-38-generic x86_64 (`uname -a`) |
| Kernel cmdline | `ro amd_iommu=off amdgpu.gttsize=126976 ttm.pages_limit=32505856 console=tty0` |
| Server container | llama.cpp runs in a podman container named `llama-server` (from `podman exec` invocation) |
| llama.cpp | router mode, build b11391-2bc563573 (b11381 earlier the same day); rebuilt by operator 2026-10-05 to b11399-2ca15f540 (§4.2), then b11403-9d3aba6b5 (§4.3), then b11430-8345f3339 (observed 2026-10-06, §4.4), `/props`: `max_instances: 1`, `models_autoload: true`, HTTP, port 9931 |
| Models configured | 8 aliases; `qwen3.8-flash-next` loaded (`load-on-startup`); others unloaded: qwen3.8-27b (preset includes MTP draft), qwen3-coder-30b-a3b, gemma-4-26b-a4b, lfm2.5-8b-a1b, mimo-v2.6-flash, qwen3-embedding-8b, qwen3-reranker-4b |
| Client box | Intel i5-9500T (6C/6T, 6 threads), 14 GiB RAM, Hermes Agent v0.21.5, no inference. LAN: ping RTT avg 0.13 ms (3 pkts, 0% loss); idle API TTFT ~0.1 s |
| Hermes wiring | provider `custom:ms-s1-max`, `chat_completions`, `http://ms-s1-max-01:9931/v1`, model `qwen3.8-flash-next`; no API key configured |

**Kernel cmdline arithmetic:** `amdgpu.gttsize=126976` (MiB) = 124 GiB. `ttm.pages_limit=32505856` × 4 KiB = 133,143,986,176 B = 124.0 GiB exactly. `amd_iommu=off` disables the IOMMU.

**Memory arithmetic:** 124 GiB GPU-visible ceiling + `cache-ram 32768` (MiB) host-side KV pool + 93.7 GB trunk exceed 128 GiB physical; actual concurrent usage was never observed at the sum. Observed with trunk loaded: `--list-devices` free = 54127 MiB → ~72 GiB of the 128000 MiB pool reported used, vs 93.67 GB (87.2 GiB) trunk file size. No explanation recorded.

## 2. Model (from live `/v1/models` meta)

Qwen3.8-Flash-Next, MoE, `n_params` 176,943,899,520, file 93,671,559,680 B, ftype `IQ4_XS - 4.25 bpw`, `n_ctx` 262144 = `n_ctx_train`, vocab 248320, `n_embd` 2560. Preset sets `swa-checkpoints 8`. mmproj (F16) loaded alongside. Active-parameter count: not reported by the endpoint.

## 3. Preset states and change timeline (measured via `/v1/models`)

| State | Settings (deltas from prior row unless noted) |
|---|---|
| **As found** (morning, b11381) | ctx 262144, parallel 4, batch 2048, ubatch 512, cache-ram 4096, cache-type-k/v f16, swa-checkpoints 8, flash-attn on, kv-unified, load-mode dio, n-gpu-layers 99, no draft |
| **Fix #1** (between 11:40 and 15:00; last as-found timeout 11:40, first observed applied when queried ~18:00) | batch 4096, ubatch 2048, parallel 2, cache-type-k/v q8_0 |
| **Fix #2** (first observed 18:58) | cache-ram 32768; also build b11381→b11391. Still: ctx 262144, no draft |
| **Fix #3 (attempted)** (2026-10-05 ~16:10) | add detached MTP draft `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` + `spec-type draft-mtp`; server aborted at init (§7.2), then reverted |

Hermes-side changes: `auxiliary.approval.timeout=120` set (was default 30 s); dummy `MS_S1_MAX_KEY` added to `~/.hermes/.env`. Both were applied after measurement M12; no gateway restart occurred during this session, so no measurement reflects them.

## 4. Measurement log

All measurements run from the client box against the live server using `./benchmark.py` (the only benchmark harness for this document). Prefill requests are non-streamed with `max_tokens: 4, temperature: 0, enable_thinking: false`; decode is a streamed greedy 200-token run with `enable_thinking: false`, and the rate is tokens / (wall time − measured TTFT). The M1–M12 rows below are retained measurements from earlier session work carried out with the same request shapes and prompt construction, so the numbers remain directly comparable to what `benchmark.py` produces.

| # | Config state | Measurement | Result |
|---|---|---|---|
| M1 | as found | Decode, 150-token streamed request | 5.66 s total (~26 tok/s) |
| M2 | as found | Prefill 12,068 tok (cold) | 29.6 s → 407 tok/s |
| M3 | as found | Prefill 48,902 tok (cold) | 136.8 s → 357 tok/s |
| M4 | as found | Prefill 122,456 tok (cold) | 561.3 s → 218 tok/s |
| M5 | Fix #1 | Decode | 8.4 s / 200 tok → 24.2 tok/s |
| M6 | Fix #1 | Prefill 12k / 49k / 122k | 27.0 s (448) / 106.6 s (459) / 328.5 s (373) tok/s |
| M7 | Fix #1 | 13.4k-token prompt, 1st / 2nd identical pass | 25.8 s (522 tok/s) / 5.3 s; 3rd pass w/ different suffix 5.1 s; new prefix control 25.6 s |
| M8 | Fix #1 | 54k prompt: cold / repeat | 131.4 s (410) / 7.9 s |
| M9 | Fix #1 | 105k prompt: cold / repeat | 214.9 s (490) / 11.2 s |
| M10 | Fix #1 | Interleaved 54k sessions A,B: A / B / A again / B again | 131.6 s / 132.0 s / 8.7 s / 8.6 s |
| M11 | Fix #2, b11391 | Decode, 200 tok | 8.4 s → 24.0 tok/s |
| M12 | Fix #2, b11391 | Prefill 7,019 tok (fresh text) | 13.5 s → 522 tok/s |

**Facts from the log:**
- Cold prefill speed rose ~1.7× at 122k after Fix #1 (M4→M6); mid-range degradation (M2→M4 −46%) became roughly flat (M6).
- Prefix-cache hits observed after Fix #1 with `cache-ram` still 4096: repeats of identical 54k and 105k prefixes returned in 7.9–11.2 s (M8–M9); a control with new text showed full cold cost (M7), confirming hits were cache effects, not warm-up.
- Two interleaved 54k sessions both retained hits (M10).
- Decode measured 24.0–26 tok/s in all three config states; batch/cache changes did not move it.

**Hypotheses not isolated (bundled fixes):** the M4 degradation mechanism (prompt-cache eviction vs other cause) and the specific Fix #1 change responsible for M6/M8–M10 (candidates: q8_0 KV, parallel 2, ubatch 2048 — applied together).

### 4.1 Fresh cold probe under Fix #2 (measured 2026-10-04 evening)

`benchmark.py` sizes its ladder with fixed per-rung seeds (4000+rung), so a second
run collides with text already in the prefix cache and returns cache-hit wall times.
(The `results/1.json` written earlier this session shows exactly that: its ladder
walls are <2 s with server prompt-eval ~570–630 ms, i.e. ~20k tok/s cache hits.)
To get genuinely cold numbers under Fix #2 — the only config state the doc had never
measured on the full ladder, since M6 was Fix #1 — `probes/probe_cold_and_decode.py`
salts the seed with the wall clock. Raw output: `probes/probe_cold_and_decode.out`,
`results/2.json`.

Cold prefill ladder (Fix #2: b11391, cache-ram 32768, q8_0 KV, parallel 2):

| prompt tokens | wall | tok/s (server pps) |
|---|---:|---:|
| 11,960  | 22.4 s  | 533 (549) |
| 49,010  | 113.7 s | 431 (434) |
| 122,300 | 420.4 s | 291 (292) |

Decode, greedy 200-token stream, as a function of prompt/context length
(server-reported `predicted_per_second`; the router returns no `usage` token counts
on streamed responses, hence server pps is the rate source):

| ~ctx tokens | TTFT | decode tok/s (server pps) |
|---:|---:|---:|
| ~8 (short) | 1.7 s  | 28.6 |
| 13k        | 23.9 s | 27.4 |
| 40k        | 88.6 s | 26.1 |
| 80k        | 226.3 s| 24.3 |

Facts added by this block: (a) the first Fix #2 long-context number, 291 tok/s at
122k, is *below* Fix #1's 373 — the M6 flattening did not persist under Fix #2, so
the 122k degradation mechanism is still not isolated (priority §2); (b) decode drops
~15% from 8-token to 80k context (28.6→24.3), i.e. context/QSA-indexer reads do
compete for bandwidth despite `swa-checkpoints 8`; (c) short-context prefill
(533 tok/s) is consistent with M12's 522, confirming the probe agrees with the doc's
own Fix #2 data at short length.

Caveat (not isolated): the three ladder rungs ran back-to-back against a shared
cache, so the 122k rung followed 61k tokens of prior KV; and the server serves a
production client, so the 7-minute 122k window could have overlapped agency traffic.
Neither was controlled, so 291 vs 373 is an observation, not a proven regression.

### 4.2 Rebuild to b11399 + full benchmark (measured 2026-10-05)

Between sessions the operator rebuilt the server. Live `/props` and `/v1/models` on
2026-10-05 report `build_info b11399-2ca15f540` (was b11391-2bc563573). Config state is
otherwise unchanged: the `qwen3.8-flash-next` preset args are exactly Fix #2 (batch 4096,
ubatch 2048, parallel 2, cache-ram 32768, cache-type-k/v q8_0, swa-checkpoints 8,
flash-attn on, kv-unified, load-mode dio, n-gpu-layers 99, no `--model-draft`, no
`--spec-type`). The 8 commits b11391→b11399, checked via the GitHub compare API
(`ahead_by 8, behind_by 0`), are all CI/docs/CUDA/x86-CPU — none touch the RADV Vulkan
decode or prefill path or qwen4exp (commit list in §7.1).

A full `benchmark.py` run under b11399 (`results/3.json`, started 17:24 CEST) gives:

Cold prefill ladder, server pps (b11391 values from `results/2.json` in brackets):

| prompt tokens | wall | tok/s (server pps) |
|---|---:|---:|
| 11,941  | 22.4 s  | 547 (533) |
| 48,992  | 114.4 s | 431 (431) |
| 122,259 | 432.3 s | 283 (291) |

Decode (greedy 200-token stream, short context; §C request is short-context only):
28.55 server pps (b11391: 28.6). Idle TTFT 0.75 s. Prefix-cache reuse at 13.4k: cold
25.9 s / repeat 5.3 s and 5.1 s / control-new-prefix 25.6 s — cache hits intact.

Facts added by this block: (a) b11399 reproduces b11391 within run-to-run noise on
every prefill rung and on short-context decode, consistent with the verified absence of
any decode/prefill code change in the delta — i.e. the rebuild is not a decode or
prefill lever; (b) the 122k cold-prefill degradation persists (283 vs 291 tok/s, both
~half the 12k rate), so the mechanism is still not isolated (priority §2); (c)
long-context decode was not re-measured this run (the §C request is short-context), so
§4.1's 24.3 tok/s @80k remains the standing long-ctx number. Same ladder caveat as §4.1:
rungs ran back-to-back against a shared cache, and the 7-minute 122k window could have
overlapped production traffic.

Hypothesis (not isolated): the Fix #1→Fix #2 drop at 122k (M6 373 → 291/283) now has
exactly two candidate causes — the `cache-ram` 4096→32768 change or the b11381→b11391
build delta (batch/ubatch/parallel/q8_0-KV were already in Fix #1). The later
b11391→b11399 build-only change left 122k flat at 283, which rules the b11391→b11399
range out but does not exonerate b11381→b11391. Neither remaining candidate can be
isolated read-only (both need operator action).

### 4.3 b11403 rebuild + MTP revert (observed via live API, 2026-10-05 ~16:12)

After the §7.2 abort the operator reverted MTP and rebuilt. Live `/props` now
reports `build_info b11403-9d3aba6b5` (was b11399); `/v1/models` shows
`qwen3.8-flash-next` back to exactly Fix #2 — no `--model-draft`, no
`--spec-type` — and `loaded`. Trunk is
`/srv/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf`
(3 shards) + `mmproj-F16.gguf`.

b11399→b11403 = 4 commits (compare API), none on the qwen4exp/MTP or RADV
Vulkan path: #29622 (batch embd+raw tokens), #29895 (server log colors),
#29435 (CUDA FA scheduling), #29633 (CUDA MMVF). Re-checked
`src/models/qwen4exp.cpp:572` at `9d3aba6b5` — the #29811 guard is still absent,
so plain MTP on flash-next still asserts on b11403.

Sibling presets that DO ship MTP on this same box, as contrast (all unloaded at
this check; different merged architectures, not qwen4exp): `qwen3.8-27b`
(`mtp-Qwen3.8-27B-Q4_0.gguf`, spec-draft-n-max 3) and `gemma-4-26b-a4b`
(`mtp-gemma-4-26B-A4B-it.gguf`, n-max 4) — consistent with §7's note that these
are qwen3/gemma4 MTP, not the flash-next (qwen4exp) path that bites #29811.

Standing behavior (observed, reproducible): `read_raw_unsafe ... Bad address` →
buffered-IO fallback appears twice on the trunk shards this load (once on the
MTP draft in §7.2). It is the `load-mode dio` read path on this box hitting
EFAULT and recovering; it recurs on every cold load and is not an MTP artifact.

### 4.4 b11430 rebuild + clean full benchmark (measured 2026-10-06)

Live `/props` on 2026-10-06 reports `build_info b11430-8345f3339` (was
b11403-9d3aba6b5); the operator rebuilt again. Config state is otherwise
unchanged — `results/4.json` §A shows the preset args still exactly Fix #2
(batch 4096, ubatch 2048, parallel 2, cache-ram 32768, q8_0 KV,
swa-checkpoints 8, flash-attn on, kv-unified, dio, n-gpu-layers 99, no
`--model-draft`, no `--spec-type`).

A full `benchmark.py` run at 16:31 (`results/4.json`, contention-canary clean):

| prompt tokens | wall | tok/s (server pps) |
|---|---:|---:|
| 11,941  | 22.3 s  | 549 |
| 48,992  | 114.2 s | 431 |
| 122,259 | 432.3 s | 283 |

Decode 28.41 tok/s server pps (b11399: 28.55; §4.1 short: 28.6), idle TTFT
0.57 s, prefix-cache reuse intact (cold 25.7 s → repeat 5.3 / 5.1 s,
control-new-prefix 25.5 s). A decode-only re-check at 17:05 gave 28.51
(`results/5.json`). b11430 reproduces b11399 within run-to-run noise on every
rung; the 122k cold degradation (283 vs ~540 at 12k) persists unchanged, so the
un-isolated 122k mechanism question (§4.2 hypothesis) is unaffected by this
rebuild.

Not isolated: the b11403→b11430 range includes ggml-org/llama.cpp#29639
("vulkan: sparse flash attention for quantized K/V", merged 2026-10-05;
Reported-by-source: +15.6% decode @64k on RDNA3/4 discrete, ~0 below 32k — not
our box). Whether the quantized-KV sparse path activates on gfx1151/RADV is
what `probes/probe_decode_sparse_fa.py` was built to discriminate; it has not
produced a valid answer yet (contention block below).

**Discarded contention runs (2026-10-06 16:54–17:36; no usable numbers).**
Three probe executions (`probes/probe_decode_vs_ctx_b11430.out`,
`probes/probe_decode_sparse_fa.{out,json}`, `probes/probe_concurrency_scaling.json`)
returned a uniform ~19.5 tok/s at *every* context depth — short included —
while bracketing clean decodes at 16:31 and 17:05 measured 28.4–28.5. The
built-in canaries flagged every sample CONTENDED (a co-tenant, not a
regression); zero clean samples were recorded, so no conclusion may be drawn
from these files. They are kept only as evidence of the contention window —
direct confirmation of the production-sharing caveat noted in §4.1. The
concurrency probe additionally exposed a harness fragility: it recorded 0
completion tokens although the server had streamed usage chunks (not
reproducible against b11430 the same evening — the corrected read path returns
correct counts). Hardened 2026-10-06; a clean re-run gated on
`probes/probe_wait_quiet.py` is still outstanding.

## 5. Hermes timeout history (from `~/.hermes/logs/errors.log`)

- 44 lines matching "timed out" total; entries include `auxiliary.approval` (30 s cap), `auxiliary.title_generation` (30 s cap), `auxiliary.kanban_decomposer` (180 s cap), all against `ms-s1-max-01:9931`. Last timeout logged 11:40:13 (title_generation), after Fix #1. Zero approval/auxiliary error or timeout lines between 12:00 and 18:00 same day (verified by log window count).
- Recurring warning: `named custom provider 'ms-s1-max' has no resolvable api_key — request will be sent with placeholder no-key-required and will 401 on auth-required endpoints`. llama.cpp server does not authenticate; endpoint accepts requests without key (all measurement runs succeeded).

## 6. radeontop capture (run on the server during agent traffic, not an isolated benchmark; values as recorded)

| Metric | Value |
|---|---|
| Graphics pipe | 98.33% |
| Shader Interpolator | 93.33% |
| Clip Rectangle | 98.33% |
| Shader Clock | 2.90 G / 2.90 G (99.84%) |
| Memory Clock | 1.00 G / 1.00 G (100.00%) |
| VRAM | 987M / 868M (113.66%) |
| GTT | 7290M / 12695M (57.42%) |
| Event Engine, Vertex Grouper+Tess, Texture Addresser/Cache, Shader Export, SIC, SMX, Scan Converter, Prim Assembly, Depth/Color Block | 0.00% |

Internal inconsistencies in this capture: VRAM denominator 868M with 113% usage; GTT total 12.4G (12695M) vs kernel `gttsize` 126976 MiB (124 GiB). Strix Halo has no discrete VRAM. Counter-name-to-hardware mapping on this chip is not validated.

## 7. MTP on Flash-Next — reported facts only

All of §7 is Reported-by-source, attributed per item; none reproduced by us.

- **PR ggml-org/llama.cpp#29030** ("qwen4exp/gemma4: gather lazy tensor rows with direct reads", pwilkin): a lazy-tensor-read prefill change (table: pp512 181→400 tok/s etc.). Title, description, and diff contain no MTP/speculative-decoding content. Status on 2026-10-04: open, unmerged, "At least 2 approving reviews required"; ngxson commented the approach is "too hacky"; ggerganov commented the problem isn't stated.
- **Discussion ggml-org/llama.cpp#28512** (drluoto, Sep 6 2026; same model Qwen3.8-Flash-Next UD-IQ4_XS, same APU 395/128 GB, Vulkan/RADV, Mesa 26.0.3). Reports: trunk without speculation ~27 tok/s; with his branch+draft: median 33 tok/s on 10 replayed agent conversations; 58.1 tok/s short-code / 30.1 prose@8k / 41.6 new-code@8k / 55.4 file-rewrite@8k (n-max 6: 63.2) / 37.8 new-code@32k / 48.5 rewrite@32k; prefill 340→510 tok/s @8k, 280→390 @32k; fresh 18k TTFT 67 s→44 s; greedy output identical. Attributes 27 tok/s trunk to memory bandwidth.
- **PR ggml-org/llama.cpp#29761** ("Qwen4Exp: add MTP", aman, reported-by-source): a *separate* MTP implementation for Qwen3.8-Flash-Next, distinct from the apepojken/drluoto NextN line tracked by #27836. Reported CUDA decode (DGX Spark, UD-IQ4_XS trunk, `--spec-type draft-mtp --spec-draft-n-max 3`): baseline 28.36 → 43.88 tok/s overall (+55%), acceptance 0.64, across 24 coding/qa/rag/writing/roleplay/etc. prompts. Merged 2026-10-01 — verified in §7.1 that this merge is already in our builds.
- **Issue ggml-org/llama.cpp#29811** (eval bug, open, reported-by-source): startup assert when running Qwen3.8-Flash-Next with a detached MTP draft (`--spec-draft-model mtp-Qwen3.8-Flash-Next-Q8_0.gguf --spec-type draft-mtp`). Repro on HIP/ROCm (Radeon AI PRO R9700), but the failing path is model-graph code (`llama_model_qwen4exp::graph_mtp` building the k-pool input when `hparams.indexer_kpool > 0`), so it is backend-agnostic. A community fix (guard `hparams.dsv4_compress_ratios[il] > 0`) is posted in the issue but not merged. Relevant to §9 Q3.
- Branch: `drluoto/llama.cpp` @ `strix-halo-vulkan`, fork of apepojken's `qwen4exp-spec-mtp` lineage. Contents reported: row-id hoisting 512 experts (upstreamed as PR #28501, +19% prefill), FR-Spec trimmed 65k draft head (+8%), `--spec-draft-p-min 0.0` (+13%), determinism fixes (`GGML_VK_DISABLE_GDN_CACHE_FUSION=1`, KV zero-on-free), tiled transpose kernel; recipe uses `-np 3 --ctx-checkpoints 8`, `-ub 2048 -b 2048`.
- Branch readme states of the FR-Spec draft head: "Needs this branch; stock llama.cpp will not load it."
- Branch readme reports ngram-mod on Vulkan hurts (64-token drafts → 200–500 ms verify), ubatch 4096 did not help, draft length >3 halves prose speed.
- Our server runs stock master b11391→b11399 (props `build_info`), which runs `spec-type draft-mtp` for the qwen3.8-27b preset (draft file present in its preset). That is a *different* merged architecture (qwen3); it is not evidence that Flash-Next MTP is in stock — the actual evidence is #29761's merge, see §7.1.
- Draft head files exist at huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF. As of 2026-09-20 the **recommended file is `mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k.gguf` (2.70 GB)**, which drluoto measured faster *and* higher-acceptance than the Q8_0 (+2.8% mean whole-stack, +8.7% prose); `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.14 GB, full vocab) remains the fallback. **We have not attempted loading any of them on b11391/b11399.**
- ROCm reversal (reported-by-source): the ~47 tok/s ROCm figure in #27950 was later abandoned; drluoto (2026-09-17) states "ROCm returns wrong answers on this chip without telling you." Treat the high ROCm numbers as not-safe-to-chase; the Vulkan/RADV path is the correct-answers path.
- Multi-slot cache retention (reported-by-source, #27950): with `-np 1`-style single-slot sharing, every concurrent session evicts the others' warm cache — the dominant *perceived* latency (1.5–4 min re-prefill at ~30k ctx), not decode. Fix is `-np 3 --ctx-checkpoints 8` (LCP routing keeps one warm slot per session; return TTFT 21–251 s → 0.3–0.5 s); checkpoints are mandatory on this arch (36 recurrent GDN layers cannot roll back without them).

### 7.1 Verification against our build (Measured via GitHub API, 2026-10-04)

Contrast to §7's Reported-only items; these are checked against the b11391 hash
`2bc563573` and PR state, so they answer §9 Q2/Q4:

- **#28501 merged 2026-09-18** and its commit `5c53396b8` is an *ancestor* of
  `2bc563573` (compare API: status `ahead`, `behind_by: 0`). The 512-expert
  row-id hoisting is therefore already in our b11391 binary — our prefill numbers
  already include that +19%; they are not drluoto's "before" numbers.
- **#29761 (qwen4exp MTP, aman) merged 2026-10-01**, merge commit
  `c061df19838ff60970faf54fd7e414953590125d`, and is an **ancestor of both b11391
  (`2bc563573`) and b11399 (`2ca15f540`)** — verified via the GitHub compare API
  (both return `behind_by: 0`). This is a *different* PR from #27836 and is the one
  that actually landed MTP for Flash-Next. **Correction** to the earlier session's
  conclusion: the plain detached MTP draft head (`--spec-draft-model … --spec-type
  draft-mtp --spec-draft-n-max …`) IS loadable on our stock builds; no branch rebuild
  is required for the *plain* (full-vocab Q8_0) draft.
- **#27836 (qwen4exp native MTP / NextN draft head)** remains open-unmerged (re-checked
  2026-10-05, last updated 2026-10-02). It is the more complete apepojken/drluoto
  NextN line and is *not* the gate for plain MTP — #29761 already landed that.
- **#29811 (MTP startup assert) still open** as of 2026-10-05; the posted
  `qwen4exp.cpp` fix is not merged. So plain MTP on this model may assert at startup
  on some configs until that fix lands.
- **#29030 (gather lazy tensor rows, prefill) open-unmerged**; **#28136 (direct
  reads for lazy PLE table) closed-unmerged**. Both prefill-only and not worth
  chasing until landed.
- **#29018 (Nemotron MTP) merged 2026-09-17** — unrelated to qwen4exp.
- **b11399 rebuild (2026-10-05) adds no decode/prefill change.** Compare API
  `2bc563573...2ca15f540` → 8 commits, all off the RADV Vulkan / qwen4exp path:
  CI `#29945` `#29954` `#29959`, docs `#29656`, chat-peg-parser `#29942`, CUDA
  `#29612` (swizzling) + `#29940` (neu_padded), x86-CPU tinyBLAS `#29806`.

Net (corrected): plain MTP speculative decoding for Flash-Next **is in stock
b11391/b11399** via #29761 (merged 2026-10-01, verified ancestor). What remains
branch-only in drluoto's `strix-halo-vulkan`: the FR-Spec trimmed-head loader
(65k-vocab draft, which stock will not load), the determinism fixes
(`GGML_VK_DISABLE_GDN_CACHE_FUSION=1` + KV zero-on-free), and the LDS pad-2 /
tiled-transpose Vulkan kernels. All were measured/reported on CUDA or another
Vulkan box, so plain MTP on *our* RADV box is untested and may hit the open #29811
startup assert. See `notes/decode-mtp-operator-recipe.md`.

### 7.2 MTP startup assert reproduced on our box (observed server log, 2026-10-05)

On 2026-10-05 ~16:10 the operator applied Recipe A step 0 to the flash-next
preset (detached draft `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, `--spec-type
draft-mtp`) and reloaded. The server aborted during `srv load_model`:

```
... I common_speculative_init_result: loading draft model '.../mtp-Qwen3.8-Flash-Next-Q8_0.gguf'
... W read_raw_unsafe: Falling back to buffered IO due to Bad address
... srv load_model: loaded multimodal model, '.../mmproj-F16.gguf'
... srv load_model: initializing, n_slots = 2, n_ctx_slot = 262144, kv_unified = 'true'
/opt/llama.cpp/ggml/src/ggml-backend.cpp:345: GGML_ASSERT(buffer) failed
```

Classification: an *observed* server event (operator's journald capture,
transcribed 2026-10-05; the referenced log file was never committed to this
repo and is not on disk), not a `benchmark.py` measurement and not a
reported-by-source quote — but the causal link to the upstream bug is verified
against b11399 source (commit `2ca15f540`).

Diagnosis: this is the open issue **#29811**, now reproduced on our RADV/Vulkan
box (the issue's only prior reproduction was HIP/ROCm, so this is a new data
point: the bug is backend-agnostic as the issue suspected). In b11399 the assert
site is the unguarded `ggml_backend_buffer_get_type` (ggml-backend.cpp:345),
reached via `ggml_backend_buffer_is_host` (which has no null check,
ggml-backend.cpp:324) ← `llama_kv_cache::set_input_k_idxs` ←
`llama_model_qwen4exp::llm_graph_input_kpool::set_input` — byte-for-byte the call
chain in #29811's posted backtrace. Root cause: `graph_mtp`'s constructor
(`src/models/qwen4exp.cpp:572`) calls `build_inp_kpool` whenever
`hparams.indexer_kpool > 0`, including for the MTP draft block, which is dense
attention (`dsv4_compress_ratios[il] == 0`, no QSA layer, no node reads the
k-pool). The resulting input tensor never gets a backend buffer, and `set_input`
dereferences NULL. The published one-line fix — guard the call with
`hparams.dsv4_compress_ratios[il] > 0` — is absent from b11399 (verified) and
still unmerged upstream as of 2026-10-05.

Two incidental notes: (a) the `read_raw_unsafe ... Bad address` line is a
separate, *recovered* warning — a `load-mode dio` read returning EFAULT that
falls back to buffered IO; it did not cause the abort and recurs on the trunk shards (§4.3). (b) The crash is the
MTP graph's k-pool assert, not the mmproj segfault posed in §9 Q3 — mmproj is
loaded on every flash-next startup and is not implicated in this failure.

Consequence for Recipe A: step 0 (plain MTP on stock) does **not** work on
b11399. Options are revert (done), wait for the #29811 fix to merge, or apply the
one-line `qwen4exp.cpp` guard in a local rebuild — see Recipe A.

## 8. Resources

- Benchmark: `./benchmark.py` — self-contained harness covering server introspection (§A), idle TTFT (§B), decode throughput (§C), cold-prefill ladder (§D), and prefix-cache reuse (§E); raw results are written to `results/N.json`. Use `--salt N` (new value per run) so §D stays genuinely cold — an unsalted rerun returns prefix-cache hits (see §4.1). See `./README.md` for usage.
- Contention gate: `./probes/probe_wait_quiet.py` — blocks until the box decodes the standard prompt at ≥26 tok/s twice in a row (exit 0); run before any long measurement (see §4.4).
- Server introspection (benchmark.py §A): `GET :9931/v1/models` (per-model presets + loaded state), `GET :9931/props`, `GET :9931/health`
- Decode request shape (benchmark.py §C): POST `/v1/chat/completions` `{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Count from 1 to 80 separated by spaces."}],"max_tokens":200,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}`
- Cold prefill + decode-vs-context probe (salted seeds): `./probes/probe_cold_and_decode.py` (output `probes/probe_cold_and_decode.out`, `results/2.json`)
- Fresh full benchmark under b11399 (2026-10-05): `results/3.json` — see §4.2. Under b11430 (2026-10-06): `results/4.json`, `results/5.json` — see §4.4.
- Sparse-FA (#29639) discriminator, decode-vs-depth with canary gating: `./probes/probe_decode_sparse_fa.py` (run so far: CONTENDED-only, §4.4; clean re-run outstanding). Predecessor: `./probes/probe_decode_vs_ctx_b11430.py`.
- Concurrency-scaling probe (aggregate decode vs slot count, §9 Q1): `./probes/probe_concurrency_scaling.py` (run so far: CONTENDED-only, §4.4; clean re-run outstanding).
- MTP draft GGUF header evidence for #29811 (4 MB GGUF prefix, parseable with `./probes/gguf_peek_local.py`; note the trailing tensor directory is truncated at the 4 MB cut): `./probes/gguf/mtp_q80_head.bin`. Range-fetch tool for other HF files: `./probes/gguf_header_peek.py`.
- Upstream research synthesis (provenance-tagged): `notes/upstream-research-2026-10-04.md`
- Operator-applied recipes (read-only server → proposed patches): `notes/decode-mtp-operator-recipe.md`

## 9. Open questions (posed, not claims)

1. Which kernels account for decode ALU occupancy, and does quant format (dequant cost) or the branch's kernel changes move decode on this box? — **Still open.** Partial: decode-vs-context measured (§4.1) shows ~15% drop 8→80k ctx, i.e. context reads are a real but minor second-order term, not the ceiling. The bandwidth-vs-ALU split is not resolved: server-side profile (perf/radeontop on the box, which needs operator access) is the missing measurement. Hypothesis recorded in `notes/upstream-research-2026-10-04.md` §6: 6B active × 4.25 bpw ≈ 3.2 GB/tok → ~80 GB/s at 25 tok/s ≈ 1/3 of LPDDR5X peak, so decode is not raw-bandwidth-saturated.
2. Does the plain (non-FR-Spec) MTP draft load on stock master b11391? — **Answered: yes (corrected 2026-10-05).** The earlier "no" was wrong: it checked only #27836 (still open) and missed #29761, which added qwen4exp MTP (`--spec-type draft-mtp` + detached draft head) and merged 2026-10-01, its merge commit being an ancestor of both b11391 and b11399 (verified, §7.1). The plain full-vocab Q8_0 draft should therefore load on stock. Open risk: the FR-Spec trimmed heads still need the branch, and the #29811 startup assert (fix unmerged) **is confirmed to fire on our box** — reproduced 2026-10-05 when the operator enabled plain MTP, see §7.2.
3. Is the MTP+mmproj segfault (reported elsewhere for other model families) applicable when mmproj is loaded but no image is sent? — **Still open.** Not testable read-only (needs the draft loaded). Adjacent evidence added 2026-10-05: #29811 reports a startup *assert* (not segfault) with MTP + detached draft on this model, in backend-agnostic graph code, with an unmerged fix — a distinct but nearby risk to the mmproj+MTP question. See §7.1. **Resolved-in-part 2026-10-05:** that assert has now been reproduced on our box (§7.2) and fires during MTP graph init before any image is sent, independent of the mmproj present in the preset. The specific mmproj-only segfault (other model families) remains untested.
4. Do the #28512 branch deltas (#28501 row-id etc.) already exist in b11391, and what remains branch-only? — **Answered.** #28501 is in b11391 (merged 2026-09-18, verified ancestor). Branch-only remaining: the FR-Spec trimmed-head loader + trim script (the plain MTP draft loader landed via #29761 and is in stock), determinism fixes (`GGML_VK_DISABLE_GDN_CACHE_FUSION=1` + KV zero-on-free), LDS pad-2 / tiled-transpose kernels. #29030 and #28136 (prefill read path) are also unmerged. See §7.1. Re-checked 2026-10-05: the b11391→b11399 delta (8 commits) contains none of these — the branch-only remainder is identical on b11399.
5. Does Hermes `reasoning_effort: high` / `compression.threshold: 0.5` interact with observed latencies beyond what §4–5 measured? — **Still open**, unexamined this session.
