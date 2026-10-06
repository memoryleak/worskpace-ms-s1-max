# Proposed operator recipes — decode throughput on ms-s1-max-01

The inference box is read-only from this workspace, so everything here is a
proposed change the operator must apply. Provenance is stated per item; nothing
below has been applied or re-measured by us.

## Correctness constraint first

Stay on Vulkan/RADV. drluoto moved off ROCm on this exact chip (Sep 17 2026):
"ROCm returns wrong answers on this chip without telling you." The headline 47 tok/s
ROCm figure (#27950) is therefore NOT a safe target. All recipes are Vulkan.

## Recipe A — the one that matters for decode (MTP speculative decoding)

Decode on stock b11391/b11399 is stuck at ~24–27 tok/s. Corrected 2026-10-05: the
earlier "rebuild required" conclusion was wrong. Upstream merged #29761 (qwen4exp MTP)
2026-10-01, and its merge commit is an ancestor of both b11391 and b11399 (verified via
the GitHub compare API) — so plain MTP speculative decoding is now in the STOCK build.
The branch rebuild below is needed only for the FR-Spec trimmed head, the determinism
fixes, and the Vulkan kernel tweaks.

Correctness caveats (both reported, unreproduced on our box): #29811 (open) reports a
startup assert running this model with a detached MTP draft, fix unmerged; and the MTP
numbers below are from CUDA or another Vulkan box, not our RADV server. Treat the first
stock attempt as an experiment with a rollback path.

Step 0 (cheapest — stock build, no rebuild). Add a detached MTP draft to the
flash-next preset and reload:

  model-draft = /srv/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q8_0.gguf
  spec-type = draft-mtp
  spec-draft-n-max = 3

Use the plain full-vocab Q8_0 head (4.14 GB); the FR-Spec `Q5_K-frspec-65k` head will
NOT load on stock (needs the branch loader).

**CONFIRMED FAILURE (2026-10-05):** this exact step was applied and the server
aborted at init with `ggml-backend.cpp:345: GGML_ASSERT(buffer) failed`. That is
the #29811 bug (open, unmerged): `graph_mtp` (qwen4exp.cpp:572) builds a k-pool
input for the dense-attention MTP block, leaving a bufferless tensor that
`llm_graph_input_kpool::set_input` dereferences during warmup. Revert works
immediately. Two ways forward, in order of preference:

1. Wait for the #29811 fix to merge into a rebuild (no code effort; timelines
   unknown — the issue is still labeled bug-unconfirmed).
2. Local one-line fix without leaving stock. Patch `src/models/qwen4exp.cpp:572`
   in a rebuild of b11399:
   ```diff
   -    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0) {
   +    if (mctx_hyb->get_idx() && hparams.indexer_kpool > 0 && hparams.dsv4_compress_ratios[il] > 0) {
   ```
   This is verbatim the fix posted in #29811 (co-authored by deepseek); it makes
   the draft skip the k-pool input on non-QSA layers, and the reporter's warmup
   completes with it. It is reported-fixed, not yet merged or re-measured on our
   box, so treat the rebuild as an experiment with the revert path ready.
3. The branch path below (Recipe A branch) — unverified whether it already
   contains this guard.

Branch path (only if step 0 fails, or to gain the +8% FR-Spec head and the
determinism/Vulkan-kernel fixes):

1. Build the Vulkan branch (fork of apepojken's `qwen4exp-spec-mtp` lineage):
   ```
   git clone -b strix-halo-vulkan https://github.com/drluoto/llama.cpp
   cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
   cmake --build build -j --target llama-server
   ```
2. Download the draft head (Q5_K is the current default, not Q8_0):
   `huggingface.co/drluoto/Qwen3.8-Flash-Next-MTP-GGUF` →
   `mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k.gguf` (2.70 GB)
3. Run (branch readme's recommended line; our box already uses `-np 2` + swa
   checkpoints, so `--ctx-checkpoints 8` is the branch analog of our
   `--swa-checkpoints 8`):
   ```
   GGML_VK_DISABLE_GDN_CACHE_FUSION=1 build/bin/llama-server \
     -m /srv/models/Qwen3.8-Flash-Next/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
     -md /srv/models/Qwen3.8-Flash-Next/mtp-Qwen3.8-Flash-Next-Q5_K-frspec-65k.gguf \
     --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.0 \
     -fa 1 -ub 2048 -b 2048 -c 262144 -np 2 --ctx-checkpoints 8 \
     -ctk f16 -ctv f16 -lm dio --jinja
   ```
   The stock UD-IQ4_XS trunk "works too and gives the prefill gain"; the best decode
   numbers use drluoto's 86 GB "AgenticRequant Q5K" requant (dense Q5_K, routers
   Q8_0, experts unchanged), requant recipe at
   github.com/drluoto/flash-next-strix-halo.

Expected (reported, unreproduced, Vulkan/RADV on this APU): trunk 27 → ~33 median on
replayed agent conversations; up to 58 (short code) / 41 (new code @8k) / 55
(rewrite @8k). Greedy output bit-identical run-to-run on the same head. Determinism
envelope: `GGML_VK_DISABLE_GDN_CACHE_FUSION=1` + KV zero-on-free are shipped in the
branch; do not omit the env var if `-np > 1`.

Not worth doing (branch author measured no gain on this hardware): ngram-mod on
Vulkan, ubatch 4096, glue-kernel fusion, a grouped expert kernel sharing weight
reads between speculative tokens.

## Recipe B — prefill (lower priority; decode dominates)

Cold prefill under our current state measures ~533 (12k) / 431 (49k) tok/s (this
session; 122k pending). Further prefill-only gains are unmerged upstream:
#29030 (direct-read gather, open, flagged "too hacky") and #28136 (closed, unmerged).
Neither is worth pursuing until they land. Our build already contains #28501.

## Recipe C — multi-client cache retention (cheap, non-kernel)

If the server ever serves more than one concurrent Hermes session, raise `parallel`
to 3 with `--ctx-checkpoints 8` (already have swa-checkpoints 8). drluoto measured
return TTFT 21–251 s → 0.3–0.5 s by giving each session its own warm slot via LCP
routing; the single-slot eviction pattern ("1.5–4 min re-prefill at ~30k ctx") is
the dominant perceived latency, independent of decode speed.

## Recipe D — GPU clock pinning

Branch/HF card note: "On AMD, pin the GPU clock first … on auto it floats and you
lose ~20%." The documented command is a ROCm tool (`rocm-smi -d 0 --setperflevel
high`); for the RADV/Vulkan stack the equivalent is unverified and needs an operator
check. Treat the ~20% as a reason to verify the clock governor during any future
bench, not as an approved action.