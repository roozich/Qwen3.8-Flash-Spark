# Configuration and checkpoints
Runtime options, checkpoint layouts and the tuning reference. Migrated from the README during the docs restructure.

## Two checkpoint modes: NVFP4 or hybrid

`scripts/serve.sh` serves one of two layouts of the same NVFP4 checkpoint (NVIDIA's by default,
RadixArk's with `MODEL=`); pick with `MODE=`. The table below was measured on RadixArk's checkpoint;
on NVIDIA's the same conversion takes decode from ~28 to ~34 tok/s and the KV pool from ~565k to ~680k.

| | `MODE=nvfp4` (default) | `MODE=hybrid` |
| --- | --- | --- |
| Routed experts (the bulk, ~63 GiB) | NVFP4 | NVFP4 (unchanged) |
| GDN in/out projections, QSA q/k/v/o, shared experts (~15 GiB) | bf16, as published | **blockwise fp8-e4m3** (128×128 blocks, DeepSeek layout) |
| Extra preparation | none | `scripts/prepare-hybrid.sh` once (~10 min, +13 GB disk) |
| Decode (MTP=2, greedy, real answers) | ~26 tok/s | **~31 tok/s (+20%)** |
| Prefill | same | same (±5%) |
| KV cache | ~580k tokens | **~630k tokens (+8%)**, weights ~7 GiB smaller |
| Tournament quality (17 agentic scenarios × 3) | 45/51 | 45/51 |
| Deterministic at T=0 | yes | yes |
| Behavioural difference we noticed | — | slightly "more careful" in tool loops: it sometimes checks state first (one extra tool call), which is the only place it scored differently before we raised the turn budget |

Why it works: those side layers are dense and read in full on every decoded token, so
they dominate decode bandwidth; the experts are sparse (10 of 512 active) and already
4-bit. Halving the dense part is where the tokens/s come from. The MoE path — where
the quality lives — is untouched. Conversion uses
[@Saren-Arterius](https://github.com/Saren-Arterius)'s `fp8_convert.py` (worst
per-tensor max relative error 3.5%), and a small dispatch shim
(`src/vllm_fp8_hybrid_modelopt.py`) that routes those layers to vLLM's blockwise-fp8
GEMM while the ModelOpt NVFP4 config keeps handling the experts.

```bash
scripts/prepare-hybrid.sh                 # builds <snapshot>-fp8hybrid/ next to the HF snapshot
MODE=hybrid YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve.sh
```

Our own box runs the hybrid. If you want the checkpoint exactly as published, stay on
`MODE=nvfp4` — you lose ~5 tok/s and nothing else.

### NVFP4 MTP draft experts (`MODE=hybrid-mtp`, RadixArk only)

**RadixArk only**: NVIDIA's checkpoint already ships its MTP drafter in fp8 (that is where its KV
advantage comes from), so there is nothing to graft and `MODE=hybrid-mtp` refuses it; the `context`
profile pins `MODEL=RadixArk/…` for that reason. A graft on top of the RadixArk hybrid: the MTP draft head's routed experts (BF16 fused, ~4.7 GiB)
are replaced by the **NVFP4** draft experts from
[Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
(1.4 GiB) — the combination neither parent ships: fp8 PLE table *and* a cheap draft.
`scripts/prepare-mtp-graft.sh` builds it in ~5 min (~3.3 GB of real bytes: one shard
rewritten without the 2 fused BF16 MTP tensors, a symlink to the donor shard, and the
index + both quantization exclusion lists fixed — the blanket `mtp.*` globs are replaced
by the donor's 29 explicit non-expert MTP module names in **both** `config.json` and
`hf_quant_config.json`, or the draft loads unquantized and dies). Both parents are
modelopt quantizations of the same base — verified by hashing `embed_tokens.weight` and
all 29 shared draft tensors byte-for-byte before grafting. The graft reads through both
parents (it is a directory of symlinks); don't delete either one.

Measured on the GX10 (hybrid, MTP=2, greedy, real answers):

| | `MODE=hybrid` | `MODE=hybrid-mtp` |
| --- | --- | --- |
| Weights on card | 77.83 GiB | **74.75 GiB** (−3.1) |
| KV cache @0.80 | 625,669 tok | **734,292 tok (+17%)** |
| Max concurrency @262k | 2.39x | **2.80x** |
| Decode, 5-prompt greedy probe | 26.6–33.8 tok/s | **34.7–42.5 tok/s (+20–27%)** |
| Mean acceptance length | ~2.73 | ~2.58 (same band) |
| Tournament quality | 45/51 | same target model, byte-identical greedy outputs (see below) |

Those are the author's numbers. Ours, measured the way defaults are decided here (same
box, same day as the table in [Reduced draft vocabulary](#reduced-draft-vocabulary-draft_vocab1-default),
full-vocabulary draft on both arms):

| | `MODE=hybrid` | `MODE=hybrid-mtp` |
| --- | --- | --- |
| Tournament (17 scenarios × 3) | 44.5/51 | **44/51** (same failures, within the day's noise) |
| tok/s in the tournament | 33.5 | 33.0 |
| Decode, single stream (bench) | 33.1 tok/s | 32.6 tok/s |
| MTP acceptance | 75% (mean length 2.50) | 63.5% (2.27) |
| KV pool @0.80, YaRN 500k | 626k tok | **764k tok (+22%)** |
| Weights on card | 77.8 GiB | **73.9 GiB** |
| Deterministic 4/4, smoke-test (cache hit + logprobs) | yes | yes |

So on this box the graft is a **memory** win, not a speed win: the NVFP4 drafter reads ~4×
fewer bytes per draft step but its proposals are accepted less often, and the two cancel.
It is quality-neutral, which is why it ships as an option rather than the default; take it
when the KV pool or concurrency matters more than the last percent. Not yet measured in
combination with the reduced draft vocabulary.

The draft swap is gated on **greedy output equivalence**: every emitted token is the
target model's argmax (the draft only decides how many of its proposals get accepted per
step), so `scripts/greedy-probe.sh <label>` run against both arms must produce
byte-identical text. It does — 5/5 prompts, 400 tokens each, first-token logprobs
identical to 4 decimals. Donor revision is pinned (`103a7608…`, sha256 `0d44e6d7…`); a
different donor revision needs re-gating.

```bash
scripts/prepare-mtp-graft.sh              # one-time, after prepare-hybrid.sh
MODE=hybrid-mtp scripts/serve.sh
```

## Optional: persistent compile cache (`COMPILE_CACHE`)

`scripts/serve.sh` recreates the container on every run (`docker rm -f`, then `docker run`), so
what vLLM and its kernels compile inside the container is discarded and rebuilt on every boot. If
you keep one container and cycle it with `./flash stop` / `./flash start` this costs nothing, which
is why it went unnoticed. It costs you when something else recreates the container for you: a proxy
that loads and evicts models on demand (llama-swap and friends), CI, a tournament run that calls
`serve.sh` between configurations, or a systemd unit around `serve.sh`.

`COMPILE_CACHE=<name>` mounts four docker volumes over the four cache directories: `<name>-vllm`
(`/root/.cache/vllm`), `<name>-flashinfer` (`/root/.cache/flashinfer`), `<name>-triton` (Triton's
`/root/.triton`) and `<name>-nv` (the CUDA driver's JIT cache, `/root/.nv`).
`COMPILE_CACHE=/some/path` binds `/some/path/{vllm,flashinfer,triton,nv}` instead, to put them on a
chosen disk. Unset — the default — is exactly the behaviour above.

Where the time goes without it (one py-spy-profiled boot):

- **Triton, ~20 s of init engine.** About 140 kernels (QSA, PLE, fused MoE, GDN, sampler) are built
  from scratch: in the memory profile, in the graph capture that sizes the KV cache (its "10 secs"
  are mostly compiles; 2 s with a warm cache), in kernel warmup and in v0.30's `warmup_kernels`.
- **CUDA driver JIT, ~5 s of init engine.** The vision tower's cuBLAS GEMMs and flash-attn hdim-96
  kernel ship as sm80 PTX, which the driver compiles for GB10 into `/root/.nv`.
- **vLLM's `modelinfos`, ~11 s of API-server startup.** Without it the model classes are inspected
  in a subprocess. v0.30 writes no torch.compile cache for this recipe, so this 16 KB is all the
  `-vllm` volume holds. The FlashInfer volume made no measurable difference.

Measured on a GX10, hybrid + YaRN 500k + MTP=2, weights over NFS, same recipe each time, one row per
boot:

| boot | init engine | graph captures | start → ready | Triton kernels / driver JIT files written |
| --- | --- | --- | --- | --- |
| unset (default) | 53.4 s | 10 + 2 s | 121.1 s | 139 / 13 |
| vllm + flashinfer volumes only | 53.5 s | 10 + 2 s | 110.3 s | 139 / 13 |
| four volumes, populating | 53.9 s | 10 + 2 s | 111.1 s | 139 / 13 |
| four volumes, reused (4 boots) | 27.5–28.1 s | 2 + 2 s | 83.4–85.0 s | 0 / 0 |
| reused, only the `-nv` volume emptied | 33.3 s | 2 + 2 s | 89.4 s | 0 / 13 |

Reused against unset is **−26 s of init engine and −37 s of start → ready** per boot. Populating
costs nothing. Disk: 51 MB Triton, 124 MB driver JIT, under 1 MB for the other two. Startup only:
greedy probe (5 prompts) byte-identical in text and first-token logprobs against a boot without the
Triton and driver volumes. Triton keys each kernel by its source, constants and backend, and the
driver by PTX and driver version, so a changed kernel or a new driver compiles again instead of
reusing a stale entry.

On the preview base, where this option started (PR #21), it was vLLM's torch.compile cache that
mattered (−80 s of init engine) and Triton's made no difference; those measurements are in
[docs/HISTORY.md](HISTORY.md#persistent-compile-cache-on-the-preview-base-pr-21).

## Reduced draft vocabulary (`DRAFT_VOCAB=1`, default)

vLLM shares the target model's `lm_head` with the MTP draft, so every draft step scores all
248,320 vocabulary rows: a 1.27 GiB bf16 read per drafted token, on a decode step that is
memory-bandwidth bound. With `DRAFT_VOCAB=1` the drafter scores a private 65,536-row slice of
the head (+320 MiB of memory) and every other token gets −∞, so the proposer's argmax/sampling
code is untouched. The target still verifies every drafted token: **outputs are identical to
full-vocabulary drafting**; only the acceptance rate can move, down, when the target wants a
token outside the set (about 6% of the tokens of a French text, 75% → 68% acceptance here).

The 65,536 ids are the most frequent tokens of a small local corpus, then BPE merge order as a
frequency proxy, plus every special and added token (chat template, tool-call and thinking
markers, byte fallbacks). `tools/build_draft_vocab.py` rebuilds the set for another language
mix; `DRAFT_VOCAB=/path/ids.npy` uses your own, `DRAFT_VOCAB=0` disables.

Measured on the GX10 the way defaults are decided here — the 17-scenario agentic tournament,
3 repeats, temperature 0.2, one variable per run, all on the same day (hybrid, MTP=2, prefix
caching, YaRN 500k):

| configuration | tournament | tok/s in the tournament |
| --- | --- | --- |
| hybrid, exact `torch.topk` (the previous default) | 43/51 (84.3%) | 31.4 |
| + deterministic kernel (PR #10) | 44/51 (86.3%) | 31.9 |
| + `MADV_RANDOM` on the table | 44.5/51 (87.3%) | 33.5 |
| **+ reduced draft vocabulary, MTP=2 (the new default)** | **45/51 (88.2%)** | **38.5** |
| same, MTP=3 | 44/51 (86.3%) | 41.2 |

Two runs of the same configuration on different days differ by up to 2 points, so the four
first rows are equivalent in quality; the last one shows why MTP=3 stays an option. Single-stream
`bench` numbers for the default: decode 36.6 tok/s, prefill 2,473 tok/s at 8k / 3,004 at 32k,
needle 92k in 45.5 s, 4/4 deterministic. The KV pool loses the 320 MiB slice plus the
full-width logits buffer the patch rebuilds per draft step (about 60k tokens at `GPU_MEM=0.80`).

## Checkpoints: NVIDIA's NVFP4 (default) and RadixArk's

Two NVFP4 quantizations of Qwen3.8-Flash-Next fit this recipe, and the recipe does not care which
one it serves: `MODEL=<org/name>` on `download-weights.sh`, `prepare-hybrid.sh` and `serve.sh` (or
`./flash setup default MODEL=…` then `./flash serve`). Both have NVFP4 routed experts, the fp8 PLE
table this whole repo is about, and bf16 side layers that `prepare-hybrid.sh` converts to blockwise fp8
the same way (the 300 side tensors are the same weights, down to the conversion error); the
deterministic top-k, the reduced draft vocabulary, prefix caching and YaRN apply identically.

- **[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** — **the
  default since 2026-09-14.** NVIDIA's own ModelOpt export (issue #17, @PathosEthosLogos). 124 GiB in
  24 files, one of them a 50 GiB PLE shard that only downloads through Xet (`download-weights.sh` uses
  Xet by default). Its MTP drafter keeps its experts in blockwise fp8 under a ModelOpt *mixed-precision*
  config; vLLM 0.29 could not load that (no method for those experts, and the layer index of the
  drafter not remapped). vLLM v0.30 loads it as is (vllm#55513); before that this repo carried the
  fix as patch 11 (a backport by @techfury90 and a stopgap shim, see [docs/HISTORY.md](HISTORY.md)).
- **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** — the
  default until 2026-09-14, the checkpoint every number in the sections above was measured on. 122 GiB
  in 418 files, plain ModelOpt NVFP4 config, bf16 MTP drafter (hence the `hybrid-mtp` graft, which only
  exists for it). Fully supported, one variable away:

```bash
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 scripts/download-weights.sh
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 scripts/prepare-hybrid.sh
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh
#   or: ./flash setup default MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 && ./flash serve default MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
```

### Why the default changed: the head-to-head (2026-09-13/14)

The rule of this repo is quality first; a default only moves on measured data, never on a smoke test.
So the two checkpoints were run through the same protocol, back to back, on the same box:

- **same image** (`qwen38-flash-dgx`, v0.29 base, the tree of `main`), **same recipe** (hybrid,
  deterministic top-k, draft vocab 65k, MTP=2, YaRN 500k, prefix caching), **only `MODEL` differs**;
- one boot per checkpoint; on that boot: speed, memory, long-context and determinism probes, then
  **5 full passes of the agentic tournament** — 55 scenarios (tool loops with injected faults,
  long ledgers, logic grids, calendar and org-chart deduction, hidden-test coding), temperature 0.2,
  32k reasoning tokens per turn, partial credit on multi-item scenarios — then a dedicated block on
  the four coding tasks where this model's reasoning most often runs away (10 samples each);
- 16 h of GPU time, 275 scored runs per side, every trajectory kept.

| | RadixArk hybrid + MTP=2 | **NVIDIA hybrid + MTP=2 (default)** |
| --- | --- | --- |
| tournament, 5 passes × 55 scenarios | 86.1% ± 1.9 (85.7 / 84.2 / 88.1 / 88.1 / 84.5) | **88.8% ± 1.0** (89.0 / 89.0 / 87.2 / 89.0 / 89.9) |
| items lost on the 46 scenarios that never run away (230 runs) | 4.1 | **0.7** |
| reasoning runaways (32k-token cap), all coding tasks | 49/65 | 46/65 (p = 0.55: equal) |
| scenarios where one side is behind | none for RadixArk beyond one-run noise on g9 | none for NVIDIA |
| needle 100k / 185k / 323k / 413k | 4/4 found | 4/4 found |
| deterministic at temperature 0 | yes | yes |
| decode, single stream, greedy prose | **37.1 tok/s** | 34.5 tok/s (−7%) |
| decode per stream, 4 concurrent agents | 21.6 tok/s | 22.1 tok/s |
| prefill 8k / 32k (cold) | 5.1 s / 10.6 s | 5.0 s / 11.3 s |
| prefill at 185k / 323k / 413k | 1,919 / 2,497 / 2,856 tok/s | 1,857 / 2,465 / 2,782 tok/s |
| KV pool @500k, `GPU_MEM=0.80` | 589k tokens | **679k tokens (+15%)**; +22% on another boot |
| weights loaded | 77.3 GiB | **74.9 GiB** |
| download | 122 GiB, 418 files, HTTPS or Xet | 124 GiB, 24 files, **Xet required** (50 GiB shard) |

How to read it, honestly:

- The 2.7-point gap is above the noise floor of a 5-pass design (bootstrap CI [+0.4, +5.3], permutation
  p = 0.048) but about half of it is luck on the runaway-prone coding tasks, where both checkpoints
  derail at the same rate. The half that is not luck is a handful of slips only RadixArk made: a
  counting error on a paginated ledger (5 of 5 passes there, 8 of 18 across three images; NVIDIA 14 of
  15), one fabricated tool trajectory, two mis-executed multi-step tool tasks. NVIDIA's only systematic
  weakness in the data is a reporting convention (it under-counts a `total_moved` line while executing
  the 30 operations perfectly).
- The claim this data supports is **"parity or better on quality, +15–22% KV, −8% single-stream
  decode, vendor checkpoint whose loading fix is upstream"** — not "NVIDIA is 2.7 points smarter".
- Under that reading the default rule is unambiguous: at quality parity or better, the checkpoint that
  leaves more room for context and concurrency, and that the next vLLM release will load with no patch
  at all, is the one to ship. If your use is a single chat stream where 37 vs 34 tok/s is what you
  feel, RadixArk keeps a small edge and is one variable away.

Derivatives of either checkpoint in the same ModelOpt layout (abliterated variants such as
`Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark` or
`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47`) run with the same commands; we do
not ship or endorse them, we only note that the recipe does not care. **Compressed-tensors derivatives**
(`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` and `lychee888/…-FP8PLE`, issue #23) are a different
format: their quantization config lives in `config.json` (no `hf_quant_config.json`, which is fine, vLLM
reads it from there) and their dense side layers are **already fp8**, so the hybrid step has nothing to
convert — `prepare-hybrid.sh` and `flash doctor` say so and point to the `published` profile
(`MODE=nvfp4`). The PLE table of the lychee888 build uses RadixArk's exact file and key layout, so the
mmap patch should apply; we have not booted one ourselves.

## How the defaults are chosen: quality first, speed as an option

The serving settings retained in `scripts/serve.sh` were selected using our **agentic
tournament** (tool loops, long-context extraction, multi-step reasoning, hidden-test coding;
17 scenarios when the recipe was built, 55 for the checkpoint decision; 3–5 repeats, temperature
0.2), run on the GX10 with one variable changed at a time, on the same day. These
measurements predate the v0.32 port; its narrower validation is documented in
[VLLM-032.md](VLLM-032.md).
A change that only buys tok/s or TTFT and costs even a point there ships as an **option**, off
by default, with its measured cost next to it. Two runs of the same configuration differ by
up to 2 points day to day, so anything inside that band is treated as equal and the faster
one wins; anything below it stays an option.

| what | default | measured effect (GX10, hybrid, MTP=2, prefix caching, YaRN 500k) |
| --- | --- | --- |
| Checkpoint (`MODEL`) | **`nvidia/Qwen3.8-Flash-Next-NVFP4`** (since 2026-09-14) | vs RadixArk, 5 × 55 scenarios: 88.8% ± 1.0 vs 86.1% ± 1.9, runaway rate equal, needle and determinism equal; +15–22% KV pool, −8% single-stream decode |
| Hybrid checkpoint (`MODE=hybrid`) | what the `flash` `default` profile runs; `serve.sh` alone defaults to `nvfp4` (as published) | +20% decode, +8% KV, same tournament score |
| Deterministic top-k kernel (`DET_TOPK=1`) | **on** | identical greedy outputs, full prefill speed, tournament neutral (44/51) |
| Reduced draft vocabulary (`DRAFT_VOCAB=1`) | **on** | +20% decode, tournament 45/51 (the best run), outputs unchanged by construction |
| `MADV_RANDOM` on the table (`MADVISE=random`) | **on** | cold prefill −4–8%, cleaner page cache, tournament neutral |
| Prefix caching (`PREFIX_CACHE=1`) | **on** | repeated prefixes skip the prefill: [details](HOW-IT-WORKS.md#prefix-caching-the-root-cause-and-the-fix) |
| `MTP=3` | option (`MTP=2` default) | +7% decode, −1 point at the tournament (44 vs 45/51) |
| NVFP4 MTP draft experts (`MODE=hybrid-mtp`) | option | +22% KV pool, −3.9 GiB weights, decode unchanged here, tournament neutral (44/51) |
| DFlash2 draft model (`DRAFT_MODEL`) | option (`dflash` profile) | +26% single-request generation throughput, more GPU memory. See [measurements](BENCHMARKS.md#dflash-profile-measurements) |
| fp8 KV cache (`KV_DTYPE=fp8_e4m3`) | option | ×1.9 KV pool, 1M context. −4% decode, −3 to −17% prefill, tournament 88.4% (3 runs, 500k context) vs 87.8% in bf16 (3 runs); prefix-cache blocks twice as coarse |
| Exact `torch.topk` (`EXACT_TOPK=1`) | fallback | deterministic like the kernel, −20–40% long prefill |
| Persistent compile cache (`COMPILE_CACHE`) | option | −37 s of startup per boot after the first (init engine −26 s); startup only, outputs and tournament unaffected |
| `concurrent` profile (`SEQS=16`, BF16 GDN state) | option | +21–94% aggregate throughput at 8–32 clients with the tested FP8 KV settings; second 55-scenario tournament pass 88.8% vs 88.9%, with one request timeout; [measurements](BENCHMARKS.md#measured-concurrent-profile-pr-54) |
| `--long-prefill-token-threshold` (via `EXTRA`) | option | keeps decoding clients responsive under concurrent prefills, at a TTFT cost |

If raw throughput matters more than the agent's reliability, the levers are `MTP=3`,
`MODE=hybrid-mtp`, the `dflash` and the `concurrent` profile: each row above carries its
measured cost. The full checkpoint-choice story (the NVIDIA vs RadixArk head-to-head, 5
tournament passes, 16 h of GPU time) is in
[the head-to-head section](#why-the-default-changed-the-head-to-head-2026-09-1314).

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
| --- | --- | --- |
| `MODE` | `nvfp4` | `hybrid` = fp8 side layers (see above; needs `scripts/prepare-hybrid.sh`). |
| `PREFIX_CACHE` | `1` | `--enable-prefix-caching`. Correct with this image (block_size fix). |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (vllm#55122): identical outputs at T=0 at full kernel speed. `0` = stock kernel (non-deterministic, may drop attention candidates, issue #3). |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (deterministic; −8% prefill at 8k, −20–40% at 32k+). Wins over `DET_TOPK` when set. |
| `DRAFT_VOCAB` | `1` | MTP drafter scores only the 65,536 most frequent tokens (+20% decode, same tournament score, outputs unchanged). `0` = full vocabulary; a path = your own ids (`tools/build_draft_vocab.py`). MTP-head only — ignored for a `DRAFT_MODEL` (dflash) draft, which requires the full verifier vocabulary. |
| `MADVISE` | `random` | `madvise` on the mmapped PLE table: `random` (no readahead: cold prefill −4–8%, cleaner page cache) or `normal`. |
| `FAST_ROWS` | `0` | PLE gathers of up to this many unique rows run inline on one thread; larger ones are split across the `WORKERS` pool. `0` sends every gather to the pool, so page faults on rows the page cache dropped overlap instead of queueing: +8% decode at 1 stream, +17% aggregate at 4 streams, same rows (see the 2026-09-14 update). `512` = the old inline fast path, faster only when every row is already cached. |
| `EFFORT_ALIAS` | `1` | Accept every `reasoning_effort` a client can send. The checkpoints' template takes only `xhigh` (default), `medium` and `low` and 400s the rest — including Claude Code's default `high`. `1` serves a copy of the checkpoint's own template whose effort-resolving line maps `high`/`max` → `xhigh` and `minimal` → `low` (other values render byte-identically; the copy goes to the first writable of `$HF_CACHE/qwen38-flash-dgx/chat-templates/`, `~/.cache/qwen38-flash-dgx/chat-templates/` and `.cache/chat-templates/` in the checkout, and is bind-mounted into the container, so a root-owned HF cache does not disable it). Applied only when the template has that check and that exact line. `0` = the template as shipped. |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | Model name exposed by the OpenAI-compatible API (`--served-model-name`). Set this to a stable client-facing name without changing the underlying Hugging Face `MODEL`. |
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `CPUSET` | `5-9,15-19` | docker `--cpuset-cpus`: pin the container to the GB10's 10 Cortex-X5 performance cores. `CPUSET=` (empty) lets a newly created container use all 20 cores. The [measured concurrent profile](BENCHMARKS.md#measured-concurrent-profile-pr-54) uses this pinning; its independent speed contribution has not been established. |
| `MAMBA_SSM_DTYPE` | | `--mamba-ssm-cache-dtype` for GDN recurrent state. Unset follows the checkpoint configuration: the tested NVIDIA checkpoint specifies `float32`. Explicit `bfloat16` halves bytes per recurrent-state element and changes its precision. Enabled by the `concurrent` profile; see [memory and quality measurements](BENCHMARKS.md#measured-concurrent-profile-pr-54). |
| `GPU_MEM` | `0.80` | Fraction of the 128 GB pool for weights+KV. `0.85` buys ~2 GiB more KV, but after a day at `0.85` the box drifted into swap, and `0.875` got OOM-killed on a 300k-token prefill with MTP. The lower you set it, the more RAM the page cache has for the 48 GiB table — which is what your prefill speed depends on (below). Right after stopping another big container the first boot can fail with "13.5 GiB KV cache is needed, larger than available" — memory not yet released; the `unless-stopped` retry succeeds. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). `3` is +7% decode but cost a point at the tournament (44 vs 45/51), so it stays an option. |
| `KV_DTYPE` | `auto` | `auto` = bf16 (recommended). `fp8_e4m3` = ~1.9× KV pool, 1M context on one box, at −10% decode / −30% prefill and a measurable quality cost — see [fp8 KV cache](HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in) before using it. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather: every gather at the default `FAST_ROWS=0`; with `FAST_ROWS=512`, only gathers above 512 unique rows (decode-sized gathers then run inline). |
| `COMPILE_CACHE` | | Keep vLLM's, FlashInfer's, Triton's and the CUDA driver's caches across boots — this script recreates the container every run, so by default they are rebuilt each time. `<name>` = four docker volumes, `/abs/path` = four bind mounts. **−37 s of startup** (init engine −26 s) per boot after the first, ~175 MB of disk; only worth setting if something recreates the container for you (a model-swapping proxy, CI, tournament runs, a systemd unit). See [above](#optional-persistent-compile-cache-compile_cache). |
| `LOG_REQUESTS` | `0` | `1` logs every prompt and output (`--enable-log-requests --enable-log-outputs`, with DEBUG on vLLM's request logger only) so `tools/vllm_watch.py` can show sessions live. Debugging only: it puts user content in the Docker log, unbounded. |
| `PROM_MULTIPROC` | `0` | `1` runs prometheus_client in multiprocess mode so engine-side metrics (`vllm:ple_mmap_*`) reach `/metrics`. Opt-in, because it stops vLLM exporting its `*_created` samples and the `process_*` / `python_*` metrics (`process_start_time_seconds` included; `vllm:ple_mmap_engine_start_time_seconds` stands in as a restart marker); see *Watching the mmapped table* below. |
| `KV_CACHE_MEM` | | Passed through as `--kv-cache-memory-bytes`. `GPU_MEM` is a fraction of *total* device memory, so it leaves whatever was already resident on the table; vLLM prints the exact figure it would accept at startup ("Replace gpu_memory_utilization config with `--kv-cache-memory=...`"). On a Spark that headroom is also what the page cache uses for the PLE table, so taking it is a trade, not free memory — watch `vllm:ple_mmap_gather_seconds_total` when you do. |
| `EXTRA` | | Extra vLLM flags, passed verbatim — e.g. `--long-prefill-token-threshold 1024` for multi-client responsiveness (see [the concurrency section](BENCHMARKS.md#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider)), `--api-key <secret>`. |

### Watching the mmapped table (`vllm:ple_mmap_*`)

The PLE table is the one component whose cost depends on runtime state rather than
configuration: how much of its 47.7 GiB the page cache is holding decides your prefill
speed, and that moves as the KV pool, the request mix and the OS all pull on the same
unified memory. The module exports five counters so this is visible on a dashboard
rather than only in a windowed log line that a container restart destroys:

```
vllm:ple_mmap_lookup_ops_total         lookups (hash + gather + H2D)
vllm:ple_mmap_op_seconds_total         cumulative seconds in the lookup op, GPU wait included
vllm:ple_mmap_gpu_wait_seconds_total   of which: waiting for GPU work queued ahead of the lookup (not PLE cost)
vllm:ple_mmap_dedup_seconds_total      of which: copying the row ids to the host and deduplicating them
vllm:ple_mmap_gather_seconds_total     of which: the row gather (disk reads)
vllm:ple_mmap_stage_seconds_total      of which: staging the rows for the GPU (pinned copy, H2D launch)
vllm:ple_mmap_rows_total               rows gathered
vllm:ple_mmap_bytes_total              bytes read from the table
```

The lookup starts with a blocking copy of the row ids to the host, and that copy waits for every GPU
kernel queued ahead of it: the n-gram hashing and the layers before the PLE layer. `op_seconds` therefore
mixes their compute with the lookup's own cost (one prefill window read 165 ms per lookup, of which 8 ms
was the gather). `gpu_wait_seconds` is that wait on its own, so `op − gpu_wait` is what the lookup itself
costs the step. The periodic `PLE mmap stats` log line shows the same split at its end:
`gpu-wait X ms/op, host Y ms/op (dedup, gather, stage)`.

They are registered in the EngineCore process, so they only reach `/metrics` when
prometheus_client runs in multiprocess mode. vLLM turns that on only for
`api_server_count > 1`; `scripts/serve.sh` does it for the single-server setup used here when you opt in with
`PROM_MULTIPROC=1`, by pointing `PROMETHEUS_MULTIPROC_DIR` at a fresh tmpfs.

Switching to multiprocess mode was checked against a live server by diffing the
complete `/metrics` before and after: vLLM's other 71 metric families are exported with
identical label sets and no per-process `pid` label. Two things are lost: the 35
`*_created` families, which prometheus_client does not export in multiprocess mode, and the
default `process_*` / `python_*` collectors (`process_start_time_seconds`,
`process_resident_memory_bytes`, `process_cpu_seconds_total`, `python_gc_*`, `python_info`),
because vLLM then serves a fresh registry that holds only the multiprocess collector (reported
by [@PhilX-rgb](https://github.com/PhilX-rgb) in [#36](https://github.com/blazux/qwen3.8-Flash-DGX/issues/36)).
For restart detection, `vllm:ple_mmap_engine_start_time_seconds` carries the EngineCore's start
time and changes on every restart. That is why exporting the counters is opt-in, so nothing
changes for existing dashboards unless you ask for it.

The views worth graphing:

```promql
# host seconds per lookup: what the PLE lookup itself costs each step
(rate(vllm:ple_mmap_op_seconds_total[5m]) - rate(vllm:ple_mmap_gpu_wait_seconds_total[5m]))
  / rate(vllm:ple_mmap_lookup_ops_total[5m])
# page-cache health: the share of the lookup's own time spent on disk
rate(vllm:ple_mmap_gather_seconds_total[5m])
  / (rate(vllm:ple_mmap_op_seconds_total[5m]) - rate(vllm:ple_mmap_gpu_wait_seconds_total[5m]))
# NVMe read bandwidth from the table
rate(vllm:ple_mmap_bytes_total[5m])
```

The disk share is the page-cache health signal. It climbs as the cache is squeezed and falls
as the hot region settles in. Divide by `op_seconds` alone and it also moves with GPU load,
which says nothing about the cache. Pair it with `Cached` from a node exporter, since nothing
in vLLM's own metrics exposes the quantity that actually governs it. `VLLM_PLE_MMAP_PROMETHEUS=0` turns the counters off.
