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
| llama.cpp | router mode, build b11391-2bc563573 (b11381 earlier the same day), `/props`: `max_instances: 1`, `models_autoload: true`, HTTP, port 9931 |
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
- Branch: `drluoto/llama.cpp` @ `strix-halo-vulkan`, fork of apepojken's `qwen4exp-spec-mtp` lineage. Contents reported: row-id hoisting 512 experts (upstreamed as PR #28501, +19% prefill), FR-Spec trimmed 65k draft head (+8%), `--spec-draft-p-min 0.0` (+13%), determinism fixes (`GGML_VK_DISABLE_GDN_CACHE_FUSION=1`, KV zero-on-free), tiled transpose kernel; recipe uses `-np 3 --ctx-checkpoints 8`, `-ub 2048 -b 2048`.
- Branch readme states of the FR-Spec draft head: "Needs this branch; stock llama.cpp will not load it."
- Branch readme reports ngram-mod on Vulkan hurts (64-token drafts → 200–500 ms verify), ubatch 4096 did not help, draft length >3 halves prose speed.
- Our server runs stock master b11391 (props `build_info`), which runs `spec-type draft-mtp` for the qwen3.8-27b preset (draft file present in its preset).
- Draft file `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` exists at huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF. **We have not attempted loading it on b11391.**

## 8. Resources

- Benchmark: `./benchmark.py` — self-contained harness covering server introspection (§A), idle TTFT (§B), decode throughput (§C), cold-prefill ladder (§D), and prefix-cache reuse (§E); raw results are written to `results/N.json`. See `./README.md` for usage.
- Server introspection (benchmark.py §A): `GET :9931/v1/models` (per-model presets + loaded state), `GET :9931/props`, `GET :9931/health`
- Decode request shape (benchmark.py §C): POST `/v1/chat/completions` `{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Count from 1 to 80 separated by spaces."}],"max_tokens":200,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}`

## 9. Open questions (posed, not claims)

1. Which kernels account for decode ALU occupancy, and does quant format (dequant cost) or the branch's kernel changes move decode on this box?
2. Does the plain (non-FR-Spec) MTP draft load on stock master b11391? (Test procedure: set `model-draft`, `spec-type=draft-mtp`, `spec-draft-n-max=3`, `spec-draft-p-min=0.0` in the flash preset, reload, watch container log.)
3. Is the MTP+mmproj segfault (reported elsewhere for other model families) applicable when mmproj is loaded but no image is sent?
4. Do the #28512 branch deltas (#28501 row-id etc.) already exist in b11391, and what remains branch-only?
5. Does Hermes `reasoning_effort: high` / `compression.threshold: 0.5` interact with observed latencies beyond what §4–5 measured?
