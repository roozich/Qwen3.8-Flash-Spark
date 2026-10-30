
<p align="center">
  <img
    src="docs/images/qwen-spark-wide.png"
    alt="Qwen running on NVIDIA RTX Spark"
  />
</p>

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

# Qwen3.8-Flash-Next on a single DGX Spark

Run **Qwen3.8-Flash-Next**, a 176B-parameter model (125B main + 51B n-gram, 6B
active), locally on a **128 GB NVIDIA DGX Spark or RTX Spark-family laptop** with
**vLLM**: NVMe-backed PLE offloading, MTP and DFlash speculative decoding, working
prefix caching, and up to **500k tokens of context**.

The catch this repo solves: the official NVFP4 checkpoint
([nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4))
does not fit next to a usable KV cache in the Spark's **128 GB unified pool**. 48 GiB
of that is the n-gram embedding ("PLE") table, a pure lookup that a token only touches
16 rows of. This repo patches the latest official vLLM v0.32.0 image to **serve that
table from NVMe via `mmap`** instead of keeping it resident. Weights drop to **~75 GiB**,
the rest of the pool goes to KV, and everything runs on stock GB10 kernels.

**Key capabilities**

- Up to **500k context** with the default profile (1M with the `context-1m` profile)
- **DFlash2 and MTP speculative decoding**, up to 57 tok/s single-stream decode
- **Prefix caching that actually works** on GB10, and deterministic greedy output in the
  documented single-request setting
- Optional **hybrid** checkpoint layout (NVFP4 experts + fp8 side layers): ~20% faster
  decode at the same measured quality
- One-command setup through `./flash`, with named profiles, a doctor, and a smoke test
- **OpenAI-compatible API** with tool calling and reasoning parsers

## Quick start

You need a 128 GB GB10 box (DGX Spark or RTX Spark-family) with ~155 GB free NVMe and
Docker; the full list is in [Requirements](#requirements), and setup problems have their
own page: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

```bash
git clone https://github.com/roozich/Qwen3.8-Flash-Spark.git && cd Qwen3.8-Flash-Spark
./flash doctor      # docker, GPU, memory, disk, port, image, weights: tells you what is missing
./flash setup       # builds the image, downloads the checkpoint (124 GiB, resumable), prepares the hybrid layout
./flash serve       # the recommended recipe (profile "default")
./flash wait        # first boot loads ~75 GiB of weights, ~3-4 min; prints the KV pool when the API is up
./flash test        # health, coherence, prefix-cache hit, determinism, tok/s
```

### Send your first request

The API is OpenAI-compatible on `http://localhost:18300/v1`, default model name
`qwen3.8-flash-next`:

```bash
curl http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

If you started the server with `SERVED_MODEL_NAME=<name>`, use that value in the request's
`model` field instead.

> **Before you expose it:** the server binds to all interfaces with **no authentication**.
> On a shared network, put it behind your gateway or pass
> `EXTRA='--api-key <secret>'`. See [Limitations & security](#limitations--security).

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~153 GB free disk** for the default setup: ~140 GB for the checkpoint plus
  ~13 GB for the hybrid layout that `./flash setup` prepares (+5 GB more for the
  NVFP4-MTP graft, optional), on top of the Docker image; on reasonably fast storage
  (the PLE table is read from it at runtime, NVMe strongly recommended; the Spark's
  onboard NVMe is ideal). `./flash doctor` checks these numbers for you.
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120, e.g. RTX PRO 6000) for testing, though this is tuned for the Spark.

## Usage: the `flash` command

`./flash` is the short path. It does not replace the scripts; it calls them with a named
set of variables, checks what is already done, and tells you what is missing.

| command | what it does |
| --- | --- |
| `./flash doctor [profile]` | checks arm64/GB10, the 128 GB pool and how much of it is free right now, docker + nvidia runtime, the port, the image, the checkpoint, the prepared layouts, disk space, and the profile itself |
| `./flash setup [profile]` | build the image if missing, download the weights, prepare the hybrid layout and the MTP graft; completed steps are skipped |
| `./flash serve [profile] [KEY=VALUE…]` | loads the profile, applies your overrides, refuses early if something is missing, then runs `scripts/serve.sh` |
| `./flash wait` | polls the container and the API, shows the loading stage, prints the KV pool when up |
| `./flash test` | smoke test against the running server (health, coherence, prefix-cache hit, determinism, tok/s) |
| `./flash status` / `logs` / `stop` / `start` / `rm` | the container's state, KV pool, active patches, running requests and prefix-cache hit rate; follow the log; stop; restart (~3-4 min); remove |
| `./flash profiles` | list the profiles below |

Other variables can be overridden anywhere: `./flash serve default MTP=3 PORT=18301`.
A variable already in your environment beats the profile, and `KEY=VALUE` arguments beat
both. Container name and port default to `qwen38-flash` and `18300`.

Everything `flash` does is also plain scripts, unchanged and still supported:

```bash
docker build -t qwen38-flash-dgx:v0.32 .         # official vLLM v0.32.0 image + the patches
scripts/download-weights.sh                      # one-time, ~124 GiB, resumable
scripts/prepare-hybrid.sh                        # one-time, ~10 min
MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh   # the recommended recipe
scripts/smoke-test.sh
```

The CPU unit tests run anywhere, no GPU:

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.32 test_ple_mmap_cpu.py
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.32 test_qsa_exact_topk_cpu.py
```

## Configuration profiles

Which profile to run (`profiles/*.env`; copy one to make your own):

| profile | use it when | trade-off |
| --- | --- | --- |
| `default` | always, unless one below fits better: hybrid checkpoint, 500k context, deterministic, MTP=2. Best tournament score (45/51, measured on the v0.30 base; not rerun on v0.32) | none, it is the reference |
| `speed` | you want a little more decode speed | +7% decode for about one tournament point (`MTP=3`) |
| `dflash` | single-request speed matters most | +26% single-request generation throughput (57 tok/s), more GPU memory |
| `concurrent` | several parallel clients | higher aggregate throughput; [measured setup](docs/BENCHMARKS.md#measured-concurrent-profile-pr-54) |
| `context` | long contexts or more concurrency | +22% KV pool, decode unchanged (`hybrid-mtp`) |
| `context-1m` | you need 1M tokens in one request | small speed cost (fp8 KV cache) |
| `shared` | several clients at once, chat responsiveness | decoding stays responsive while others prefill; single-stream TTFT −17–36% |
| `published` | you want the checkpoint exactly as published | nothing to prepare; ~26 tok/s |
| `native` | you never go past the native 262k context | no YaRN rope scaling |

The `dflash` profile needs one extra one-time download, the ~1.2 GiB DFlash2 draft model
(`serve dflash` checks for it and tells you if it is missing):

```bash
./flash setup dflash   # one-time: downloads the draft model (scripts/download-draft.sh)
./flash serve dflash
```

Note on two similar words: the **default profile** runs the **hybrid** layout
(faster, same quality); `serve.sh` on its own defaults to `MODE=nvfp4`, the published
checkpoint. With `flash`, hybrid is the default.

Every `serve.sh` environment variable, the checkpoint choices, and how every default was
picked (the agentic tournament, with each option's measured cost) are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md#how-the-defaults-are-chosen-quality-first-speed-as-an-option).

## Performance

Results were measured on DGX Spark / GX10 class GB10 hardware with the default
settings. Some measurements are from earlier vLLM versions; see
[docs/BENCHMARKS.md](docs/BENCHMARKS.md) and
[docs/VLLM-032.md](docs/VLLM-032.md) for methodology and current validation
coverage.

| | value |
| --- | --- |
| Single-stream decode | 45.2 tok/s (MTP=2) / **57.0 tok/s** (`dflash`) |
| Prefill | ~2,700–4,000 tok/s (8k–32k prompts) |
| Context | ~505–520k-token KV pool (YaRN 500k); 1M with fp8 KV |
| Prefix-cache hit | TTFT ~14 s → ~1.4 s on a 20k-token prefix |
| Cold boot to API | ~2 min 8 s (weight-loading patches 14–18, default on) |

## What this fixes on GB10

Three things were broken for this model on GB10 and are fixed in the image, the first
two on by default:

- **Prefix caching** used to crash, and then to silently restore an all-zero Mamba state
  on cache hits (a vLLM block-size mix-up; a two-line fix). Cold and cache-hit outputs are
  bit-identical; on a 20k-token prefix, TTFT goes from ~14 s to ~1.4 s.
  [Root cause and fix](docs/HOW-IT-WORKS.md#prefix-caching-the-root-cause-and-the-fix).
- **Non-deterministic top-k** in the sparse attention (a GB10 kernel that can drop
  candidates). The image ships a deterministic kernel rewrite (upstream
  [vllm#55122](https://github.com/vllm-project/vllm/pull/55122)): repeated requests give
  byte-identical greedy output at full speed.
  [Details](docs/HOW-IT-WORKS.md#exact-top-k-for-the-sparse-attention).
- **DFlash speculative decoding was refused** for this model: vLLM's guard rejected the
  architecture outright. The image opens it (allowlist + a target aux-hidden-state seam,
  per [vllm#56088](https://github.com/vllm-project/vllm/pull/56088)), which is what makes
  the `dflash` profile's **57 tok/s** single-stream decode possible. Opt-in:
  `./flash setup dflash` once, then `./flash serve dflash`.

- Boot used to be ~11 min of "Loading weights": five default-on loader patches take
  weight loading from 450–541 s to **35.5 s**, startup to **2 min 8 s**.
  [Details](docs/HOW-IT-WORKS.md#weight-loading-the-per-expert-h2d-copy-patch-14).

## How it works (the one idea)

A token's n-gram lookup reads **16 rows × 160 bytes ≈ 2.5 KB**. Over a 20k-token prefill
that's ~1.3 GB of small reads, under a second on NVMe, and the hot n-grams stay in the
page cache. So the 48 GiB table never needs to be in the unified pool: we `mmap` the
checkpoint's `model-plefp8-*.safetensors` shards and gather rows on demand. Nothing else
about the model changes: the hashing, dequant, and the sparse attention all run stock.
Everything else (all 23 patches, the GB10 bugs, the native-offload alternative, the
long-context findings) is in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

## Limitations & security

- **No authentication, binds all interfaces.** `scripts/serve.sh` publishes the API on
  `0.0.0.0:$PORT` with no key. On a shared network put it behind
  your gateway or pass `EXTRA='--api-key <secret>'` (vLLM then requires it as a Bearer token).
- **One big model at a time.** At `GPU_MEM=0.80` this uses most of the 128 GB pool;
  don't co-locate another large model. More: [TROUBLESHOOTING](docs/TROUBLESHOOTING.md#booting-and-memory).
- **No `torch.compile` for this model.** vLLM v0.30+ captures it with *breakable*
  piecewise CUDA graphs; the PLE lookup ends a graph segment by itself
  ([HOW-IT-WORKS](docs/HOW-IT-WORKS.md)).
- **1M context** needs the fp8 KV cache (profile `context-1m`), which costs some speed;
  in bf16 a single 1M request needs ~26 GiB of KV and 500k with YaRN is the validated
  ceiling.
- **Exact top-k costs prefill** on long prompts (`EXACT_TOPK=1` is the fallback path, not
  the default kernel). A fused kernel would recover most of it; PRs welcome.

## Documentation

| You want | Go to |
| --- | --- |
| What changed in each release | [CHANGELOG.md](CHANGELOG.md) |
| Which local patches have an upstream vLLM issue/PR, and its live status | [docs/UPSTREAM-TRACKING.md](docs/UPSTREAM-TRACKING.md) |
| Setup failures, download issues, OOM, common errors | [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| Every `serve.sh` env var, profile choice, how defaults were chosen, checkpoint modes | [docs/CONFIGURATION.md](docs/CONFIGURATION.md) |
| How each patch works internally (mmap PLE, block_size fix, top-k kernel, weight loading) | [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md) |
| Throughput tables, DFlash vs MTP measurements, benchmarking method | [docs/BENCHMARKS.md](docs/BENCHMARKS.md) |
| Parser / tool-call / Responses fixes and their validation scope | [docs/PARSER-FIXES.md](docs/PARSER-FIXES.md) |
| Parser safety design | [docs/SAFE-OUTPUT.md](docs/SAFE-OUTPUT.md) |
| The full per-file repo map (every patch number, and the CPU unit tests) | [docs/REPO-MAP.md](docs/REPO-MAP.md) |
| v0.31 / v0.32 base migration and validation | [docs/VLLM-031.md](docs/VLLM-031.md), [docs/VLLM-032.md](docs/VLLM-032.md) |
| Who contributed what, and when | [docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md) |
| What older releases said (preview image, v0.29/v0.30 bases) | [docs/HISTORY.md](docs/HISTORY.md) |

**Current base:** official vLLM **v0.32.0** (since 2026-10-22).

## Contributing

Issues and PRs are welcome: this recipe is built entirely from them, and the
[Credits](#credits) list is the proof. Good first steps:

- **Bug reports**: include your box (DGX Spark / GX10 / other), the image tag or
  `git rev-parse HEAD`, the profile and any `KEY=VALUE` overrides, and the container log
  around the failure. `./flash doctor` output helps a lot.
- **Patches to the vLLM tree**: every change here is a standalone patch under
  `src/` or `src/patches/` with an env-gated switch and, where possible, a CPU
  regression test (`src/test_*_cpu.py`). Upstream-first: if a fix can go to vLLM,
  please link the upstream PR.
- **Measurements**: follow the hygiene rules in
  [docs/BENCHMARKS.md](docs/BENCHMARKS.md#benchmarking) (idle box, boot-to-boot
  variance band, `finish_reason` checks) so numbers can be compared across PRs.

## Credits

This recipe is built entirely from community issues and PRs. The full contributor
history (who landed which fix, and when) is in
[docs/CONTRIBUTORS.md](docs/CONTRIBUTORS.md); per-release details are in
[CHANGELOG.md](CHANGELOG.md).

<p align="center">
  <a href="https://github.com/roozich/Qwen3.8-Flash-Spark/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=roozich/Qwen3.8-Flash-Spark" alt="Contributors" />
  </a>
</p>

Foundations:

- Origin: this repo began as a fork of **[@blazux](https://github.com/blazux)**'s
  [qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX); it has since developed
  into an independent project with its own image, patches, profiles and roadmap.
- The two ideas behind the reduced draft vocabulary and the `MADV_RANDOM` table advice come
  from **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)**'s
  recipe for their own checkpoint; both are reimplemented here from scratch (their code is
  AGPL-3.0) and measured on this checkpoint.
- GB10 FLA fixes, the faster PLE gather, the state-copy guard and the fp8 side-layer
  conversion: **[@Saren-Arterius](https://github.com/Saren-Arterius)**
  ([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)).
- Model: **Qwen team, Alibaba** (Qwen3.8-Flash-Next).
- NVFP4 checkpoints: **[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** (the default since 2026-09-14) and **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** (the default before, still supported).
- NVFP4 MTP draft experts (the `hybrid-mtp` graft donor): **[Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)**; the graft recipe follows
  [thavoc's graft write-up](https://gist.github.com/thavoc/d7083457f6f2d981f879670c34df34ab)
  and [Peuqui/mtp-quant-transplant](https://github.com/Peuqui/mtp-quant-transplant).
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:v0.32.0`; the model's support started as
  the `release/qwen38next` recipe / PR #53896); the Mamba state-copy race fix is
  [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) by **@AndreasKaratzas**.
- The mmap-PLE patch, the hybrid dispatch for ModelOpt-NVFP4, the prefix-caching root
  cause and fix, the exact top-k path and the GB10 serving recipe in this repo: see
  [LICENSE](LICENSE) (Apache-2.0).

## License

Apache License 2.0; see [LICENSE](LICENSE). The model weights are **not** included
and carry Qwen's own license (with a MAU/revenue clause); review it before
production use.
