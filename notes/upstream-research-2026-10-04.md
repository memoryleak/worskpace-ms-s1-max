# Upstream research — Qwen3.8-Flash-Next on Strix Halo (Vulkan/RADV)

Gathered 2026-10-04 from public sources. Taxonomy as in strix-halo-qwen3.8-perf.md:
Measured (this session) / Reported-by-source (quoted from cited source) / Hypothesis
(causal claim not isolated). Everything below that is not a direct `benchmark.py` /
probe measurement is Reported-by-source with its source named.

> **Correction 2026-10-05 (Measured, GitHub API).** §3 below concluded stock b11391
> cannot run MTP for Flash-Next because #27836 is unmerged. That was wrong: a separate
> PR #29761 ("Qwen4Exp: add MTP") merged 2026-10-01 (commit `c061df19`) and is an
> ancestor of both b11391 and b11399. Plain MTP is in stock; only the FR-Spec trimmed
> head and the Vulkan determinism/kernel fixes remain branch-only. See §7.1 of the
> perf doc. Open risk: #29811 startup assert with MTP on this model, fix unmerged.

## 1. The model (config) — Reported-by-source

From the ollama library card, the vLLM recipe (recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next),
the omlx issue #3170 config dump, and the IntuitionLabs architecture note. All agree:

- architecture `qwen4_exp`, "Qwen4ExpForConditionalGeneration"
- 48 layers: 36 linear_attention (Gated DeltaNet, GDN) + 12 full_attention
  (`full_attention_interval: 4`)
- hidden 2560, head_dim 256, 24 Q heads / 2 KV heads
- MoE: 512 experts total, 10 routed + 1 shared active per token
- Parameters: 125B backbone + 51B n-gram embedding (≈20M bigram/trigram entries,
  layer 2, host-RAM offloadable) + 4B MTP head ≈ 180B total
- Active compute: 6B params per token (backbone); n-gram lookup reads a handful of
  rows, not the whole 51B table
- n_ctx 262144 (extensible to 1M+)

The doc §2 n_params 176.94B from the endpoint matches 125B + ~51.9B (backbone +
n-gram), i.e. the trunk GGUF excludes the 4B MTP head.

## 2. What our build already has — Measured (git history, this session)

Verified via the GitHub compare API against build hash `2bc563573` (b11391):

- **#28501 merged** (2026-09-18): "vulkan: raise the hoisted row-id limit for
  mul_mat_id from 256 to 512 experts". `5c53396b8` is an ancestor of `2bc563573`
  (compare status "ahead", behind_by 0). => the +19% prefill hoisting found in the
  drluoto branch is ALREADY in our stock b11391 binary. Our M6/M12 prefill numbers
  (373–522 tok/s) already include this; they are NOT the "before" (340@8k / 280@32k)
  numbers drluoto quotes for the slow path.
- **#29018 merged** (2026-09-17): Nemotron MTP support. Not relevant to qwen4exp.

## 3. What our build does NOT have — Measured (git history / PR state)

- **#27836 (qwen4exp native MTP / NextN draft head, `--spec-type draft-mtp`) is NOT
  merged** (`merged: false`, no merge_commit_sha). This is the whole reason stock
  b11391 cannot run MTP speculation for Flash-Next. It also explains why the
  server's qwen3.8-27b preset runs `spec-type draft-mtp` (a *different* merged arch,
  qwen3) while Flash-Next does not.
  => §9 Q2 is answered: the plain MTP draft will NOT load/run on stock master b11391.
  The draft head is a *detached* head (`-md`) whose loader support lives in the
  unmerged #27836; drluoto's HF card states "Loading a detached draft head needs a
  build that supports it. Still under review upstream."
- **#29030 open, not merged**: "qwen4exp/gemma4: gather lazy tensor rows with direct
  reads" — prefill-only, still under review ("too hacky" per ngxson).
- **#28136 closed, not merged**: "qwen4exp: direct reads for the lazy PLE table
  (>2x prefill on GB10)". Closed without merge.
- **#28243 closed, not merged**: "models: Qwen3.8-Flash-Next MTP".

Net for decode: on stock b11391 there is NO merged decode improvement available.
The only decode levers all require a rebuild of the drluoto branch (or waiting for
#27836 to land upstream).

## 4. Decode landscape on this exact box — Reported-by-source

All from drluoto, Bosgame M5 = Ryzen AI Max+ 395 / Radeon 8060S / 128 GiB,
Vulkan/RADV, Mesa 26.0.3, same model & UD-IQ4_XS trunk. This is the closest
published setup to ours (ours: Mesa 26.0.8, RADV, 128 GiB, UD-IQ4_XS).

Trunk without speculation ≈ 27 tok/s; drluoto attributes this to "the memory
bandwidth of this box."

Discussion #28512 + branch readme decode table (single stream, greedy, cold cache),
decoded tok/s:

| workload | original (Sep 4) | no speculation | this branch | branch n-max 6 |
|---|---:|---:|---:|---:|
| short code, no ctx | 49.0 | 32.8 | 58.1 | — |
| new code @8k | 30.7 | 29.6 | 41.6 | — |
| prose @8k | 23.6 | 29.3 | 30.1 | — |
| file rewrite @8k | 30.9 | 29.0 | 55.4 | 63.2 |
| new code @32k | 34.5 | 25.1 | 37.8 | — |
| file rewrite @32k | 37.4 | 25.1 | 48.5 | 55.2 |

Prefill 340→510 tok/s @8k, 280→390 @32k. Ten replayed agent conversations: median
25→33 tok/s; fresh 18k TTFT 67s→44s. Greedy output bit-identical.

The 5 levers, in effect order (branch readme "in order of effect"):
1. #28501 row-id hoisting 512 experts (+19% prefill) — already in our build.
2. FR-Spec trimmed 65k-vocab draft head (+8% decode; draft LM head was 81% of the
   bytes read per drafted token).
3. `--spec-draft-p-min 0.0` always draft 3 tokens (+13% decode; extra draft tok ≈4ms).
4. Determinism fixes: KV zero-on-free + `GGML_VK_DISABLE_GDN_CACHE_FUSION=1`.
5. Small: LDS pad-2 for coopmat tiles (RADV>=25.3), tiled transpose kernel for the
   delta-net conv concat, `-ub 2048`.

Tried and did NOT help (do not repeat): glue-kernel fusion (dispatch count not the
bottleneck), grouped expert kernel to share weight reads between speculative tokens
(Infinity Cache already does that), ubatch 4096, ngram-mod on Vulkan (64-token
drafts → 200–500 ms per verify).

### Draft head (HF drluoto/Qwen3.8-Flash-Next-MTP-GGUF) — Reported-by-source

Detached MTP head, 31 draft tensors + shared embeddings + lm_head, extracted by
range-read (7.3 GB of the 360 GB checkpoint). Files (Sep 20 "new default is Q5_K"):

- `mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k.gguf` 2.70 GB  ← current default
- `mtp-Qwen3.8-Flash-Next-Q8_0-frspec-65k.gguf` 3.64 GB
- `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` 4.14 GB full vocab (doc §7/§9 currently cite this)
- `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` 2.79 GB (tight memory)
- `mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k-bf16path.gguf` 2.70 GB (built straight from bf16)

Q5_K vs Q8_0 whole-stack decode tok/s (86 GB Q5_K trunk, n-max 3, p-min 0.0, clock
pinned, greedy): mean 46.5 vs 45.2 (+2.8%); prose 31.9 vs 29.4 (+8.7%); acceptance
0.89 vs 0.86 short-code, 0.39 vs 0.35 prose. A draft matched to the target beats a
higher-bpw one. Draft choice changes which tokens verify together — output
deterministic run-to-run but NOT byte-identical across head choices (same answers).

### ROCm vs Vulkan — Reported-by-source (governance-relevant)

#27950 (Aug, ROCm 7.1) reported 47.1 tok/s file-rewrite@8k via hipCUB/radix TOP_K +
native MTP. **drluoto later moved OFF ROCm (Sep 17): "ROCm returns wrong answers on
this chip without telling you."** Do not switch the server to ROCm to chase the 47
number; the Vulkan path is the correct-answers path. The earlier ROCm gain driver
(GPU TOP_K falling back to CPU past ne=1024) is a ROCm-only defect and does not apply
to our RADV Vulkan stack.

## 5. Multi-slot finding (relevant to our production Hermes use) — Reported-by-source

drluoto #27950: one slot shared by cron + delegations means every session evicts the
others' cache; "coming back to the main chat meant 1.5–4 min re-prefill at ~30k ctx…
that is what 'slow' actually felt like — not decode." Fix: `-np 3 --ctx-checkpoints 8`;
the router already routes each request to the most-similar slot (LCP), so each
session keeps its own warm cache; return TTFT 21–251s → 0.3–0.5s. Checkpoints are
mandatory on this arch (36 recurrent GDN layers can't roll back without them).
Slot save/restore to disk is NOT a working alternative (re-prefills from zero anyway).

Our server already runs `-np 2 --swa-checkpoints 8`; the finding suggests `-np 3`
could help a multi-client Hermes deployment.

## 6. Hypotheses (ours, not isolated)

*Decode is not at raw box bandwidth.* 6B active params × 4.25 bpw ≈ 3.2 GB/token.
At 24–27 tok/s that is ~77–86 GB/s effective, ≈30–34% of the ~256 GB/s LPDDR5X-8000
(256-bit) peak for this APU. If decode were purely weight-read-bandwidth-bound it
would sit far higher; the gap implies substantial per-token overhead (Vulkan dispatch/
fence latency, on-GPU dequant, MoE routing + QSA indexer scans) or an achieved
bandwidth for this gather pattern well below peak. The "27 tok/s = memory bandwidth"
attribution from the thread is therefore best read as "per-pass floor", not raw
DRAM saturation — consistent with MTP helping by cutting the *number of sequential
trunk passes* (3 tokens verified per pass) rather than bytes read.

These remain hypotheses until a server-side profile (perf / radeontop during a
bounded decode, which requires operator access) confirms the split.