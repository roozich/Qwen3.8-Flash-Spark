# Performance and benchmarking
Throughput, concurrency and benchmarking methodology for this recipe on GB10. All figures were measured on our own boxes; measurement hygiene rules are at the end. Migrated from the README during the docs restructure.

## Throughput and concurrency

Single-stream numbers understate this model on a GB10. @jschmied traced one box
under load (RadixArk NVFP4, 8k ctx, **no** speculative decoding, using vLLM's native
PLE CPU offload rather than this repo's mmap — the table-serving cost behaves the
same way) and found aggregate throughput scales far past single-stream:

| concurrent streams | aggregate tok/s | per stream | major faults / token | TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 17.1 | 17.1 | 16.0 | 0.22 s |
| 8 | 87.5 | 10.9 | 7.0 | 0.53 s |
| 16 | 131.6 | 8.2 | 9.6 | 0.83 s |
| 32 | 212.0 | 6.6 | 4.3 | 1.19 s |
| 48 | **266.8** | 5.6 | 3.6 | 1.60 s |

Two things worth knowing (their words, lightly condensed):

- **The paged table is an argument *for* concurrency, not against it.** Page-fault cost
  per token *falls* 4.4× from c=1 to c=48: batched tokens share n-gram rows and the
  page cache keeps the hot set, so the marginal token is far cheaper than the first.
  The table gather itself never exceeded ~25% of one CPU core.
- **A low `--max-num-seqs` is indistinguishable from saturation if you only look at
  tok/s.** With `--max-num-seqs 2` their sweep flatlined at ~33 tok/s while
  `vllm:request_queue_time_seconds_sum` climbed to 142 s. Check `max-num-seqs` before
  quoting an aggregate number — this repo's default is now `8` for that reason.

Method and harness: [load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md).

### Decoding clients stall while other clients prefill (the `long-prefill-token-threshold` slider)

Reported by [@kutovoy](https://github.com/kutovoy) in
[issue #9](https://github.com/blazux/qwen3.8-Flash-DGX/issues/9): with two or more agents on
the box, a client that is decoding drops from 30 tok/s to 0.1–0.5 tok/s for a minute or two,
then recovers. Reproduced here in minutes and it is not a bug in the recipe: vLLM runs one
step at a time, and with chunked prefill every step that carries a prefill chunk also carries
exactly one token for each decoding request. On this model a chunk of 8,192 tokens
(`--max-num-batched-tokens 8192`, the default) takes ~3.5 s of compute at ~2,400 tok/s, plus
the n-gram lookups of a prompt the page cache has never seen (0.3–0.4 s per chunk here, more
on a box that is short on cache or swapping). So while any new prompt is being prefilled, a
decoding client gets one token per step, i.e. one every 4–7 s. Prefix caching does not help:
the prompts are new.

The knob is `--long-prefill-token-threshold N` (per-request chunk cap; the total step budget
stays at 8,192 for the decoders), passed through `EXTRA=`. Measured on the GX10 (hybrid,
MTP=2, prefix caching): one client decoding, then two other clients sending cold ~72k-token
prompts 10 s later.

| `--long-prefill-token-threshold` | decoding client during the two prefills | p95 / max gap between its tokens | TTFT of each 72k prompt | single-stream prefill 8k / 32k | needle 92k |
| --- | --- | --- | --- | --- | --- |
| none (8,192 chunks, default) | **0.2 tok/s** | 5.5 s / 7.1 s | 66 s / 110 s | 2,493 / 2,994 tok/s | 46.6 s |
| 2048 | 0.4–0.6 tok/s | 1.9 s / 3.0 s | 89 s / 89 s | not measured | — |
| 1024 | 1.0 tok/s | 1.3 s / 2.0 s | 98 s / 98 s | 1,597 / 2,497 tok/s (−36% / −17%) | 64.8 s |
| 512 | 1.6–1.8 tok/s | 0.85 s / 1.5 s | 112 s / 112 s | 1,268 / 2,044 tok/s (−49% / −32%) | 84.8 s |

Read it as a slider, not a fix: on one GPU, keeping a decoding client at X tok/s while
others prefill means at most ~2,400 / X prefill tokens per step, and every step below 8,192
tokens costs single-stream prefill (a 1,024-token cap already takes 36% off an 8k TTFT). A step
still holds one chunk of *each* running prefill, which is why 512 does not reach 5 tok/s. The
smaller chunks do have one unambiguous benefit: peak swap-out during the prefills fell from
90–120 MB/s to 7–60 MB/s, because the activation peak per step shrinks.

- Single main user, occasional second client (our case): keep the default. Long prompts land
  fast; the rare overlap costs the other client a slow minute.
- Several interactive agents that must stay responsive: `EXTRA='--long-prefill-token-threshold 1024'`
  (or `512` if TTFT matters less than never stalling). Confirmed in the field by
  [@techfury90](https://github.com/techfury90) with 2–6 parallel agents. Keep swap small
  (`vm.swappiness=10` — the Spark default is 60; a 134 GB swap file lets the kernel page vLLM
  itself out instead of dropping cache, and once it is swapped every step page-faults) and use
  `PREWARM=1`. Lower `GPU_MEM` (0.75) only if the box is actually swapping: with several
  long-context agents the KV pool matters more than page cache for the table. `SEQS=4` limits
  how many prefills can interleave.
- The structural way out is a second Spark: the four ConnectX-7 ports exist for that, and
  vLLM's prefill/decode disaggregation puts the prefills on the other box.

### Measured concurrent profile (PR #54)

[PR #54](https://github.com/blazux/qwen3.8-Flash-DGX/pull/54), contributed by
[@aphex3k](https://github.com/aphex3k), adds `CPUSET`, `MAMBA_SSM_DTYPE`, the
`concurrent` profile and the `bench/` performance, long-context and HumanEval tools.
CPU pinning defaults to `5-9,15-19` for all profiles; `CPUSET=` disables it when
creating the container. `concurrent` additionally selects `SEQS=16` and BF16 GDN
state, while the `default` profile keeps `SEQS=8` and the checkpoint's state dtype.
Run `./flash serve concurrent` to select it.

Measured on a GX10 / GB10 on 2026-10-13–14, with vLLM v0.31, the NVIDIA hybrid
checkpoint, MTP=2, YaRN 500k, prefix caching, `PREWARM=1`, **FP8 KV and a fixed
8 GiB KV pool**. The baseline used all 20 CPUs, `SEQS=8` and FP32 recurrent state;
the concurrent configuration used `CPUSET=5-9,15-19`, `SEQS=16` and BF16 state.
These three changes were measured together. Settings matching this comparison:

```bash
./flash serve concurrent KV_DTYPE=fp8_e4m3 KV_CACHE_MEM=8589934592 PREWARM=1
```

| Simultaneous requests | Baseline tok/s | Concurrent tok/s | Change |
| --- | ---: | ---: | ---: |
| 1 | 46.3 | 47.2 | +2.1% |
| 8 | 185.5 | **225.3** | **+21.5%** |
| 16 | 186.8 | **362.1** | **+93.8%** |
| 32 | 180.7 | **325.5** | **+80.2%** |

Throughput is total completion tokens divided by client wall time, including
request overhead. Single-request values are medians of six samples; parallel
values are medians of two sweeps. The speed probes used short prompts, output caps
of 400 tokens for single requests and 256 for parallel requests, and
`chat_template_kwargs={"enable_thinking": false}`. Ollama and LiteLLM remained
resident; baseline gateway traffic was not blocked. The small single-request
difference is not a convincing speedup. At 16 clients, median time to first token
fell from **5.90 s to 0.52 s**, with less queueing at the higher sequence cap.
The CPUSET-only contribution remains inconclusive.

**Prefill:** no clear gain. Time to first token for fresh ~8.5k / 34k / 101k-token
prompts was 3.04 / 11.09 / 33.13 s in the baseline and 3.37 / 11.01 / 32.99 s with
the concurrent configuration; all had `cached_tokens=0`. A 479,980-token retrieval
probe succeeded in both configurations in **175.4 s**. The fixed 8 GiB cache
accommodated **573,208 tokens vs 542,857 (+5.6%)** with BF16 recurrent state; this
is a capacity gain, not a doubling of the whole KV pool.

**Agentic quality:** two passes of the same 55-scenario tournament per configuration,
temperature 0.2, four parallel scenarios and a 32,000-token response cap:

| Tournament pass | Baseline | Concurrent |
| --- | ---: | ---: |
| First | 89.0% | 85.4% |
| Second | 88.9% | 88.8% |
| Two-pass mean | 88.95% | 87.1% |

The second baseline pass had six token-limit finishes; concurrent had five and
one request timeout after 1,800 s (`g21_code_regex`, scored zero and included above).
The two extra first-pass failures, `g9_code_topo` and `g36_code_justify`, both passed
in the second concurrent run. No server crash was observed. Two unseeded passes
do not establish equal quality or rule out a regression; the results support
using this profile for aggregate throughput, with the precision change and
request timeout included in its measured tradeoffs. Other KV dtypes and pool
sizes were not covered by this comparison.

### Tuning findings cross-referenced from the 27B sibling recipe

[aphex3k/Qwen3.8-27B-on-DGX-Spark](https://github.com/aphex3k/Qwen3.8-27B-on-DGX-Spark)
serves the same Qwen3.8 hybrid architecture (GDN + PLE) on this same box, but through
SGLang instead of vLLM. Its [measured results](https://github.com/aphex3k/Qwen3.8-27B-on-DGX-Spark/blob/main/results/RESULTS.md)
motivated the tuning knobs ported here. The section above reports measurements on
this model with vLLM; the findings below distinguish the shared ideas from
SGLang-specific settings.

**Ported (this repo):**

- **Concurrency headroom** — the sibling sweep showed aggregate throughput tracks the
  running-request cap, not the hardware: at the shipped cap of 4, 4 → 8 streams gained
  only 2%; at cap 16 the peak reached 480.7 tok/s (vs 190.1 at cap 8). The equivalent
  knob here is `SEQS` (`--max-num-seqs`): the `concurrent` profile raises it to 16 for
  aggregate-throughput runs. This matches what the @jschmied sweep above already showed
  for this model.
- **`MAMBA_SSM_DTYPE=bfloat16`** — the sibling's `--mamba-ssm-dtype bfloat16` halves the
  GDN state pool (30.9 GB → 15.6 GB at pool 80; float32 is the engine default). The vLLM
  equivalent, `--mamba-ssm-cache-dtype`, is exposed as `MAMBA_SSM_DTYPE` and set by the
  `concurrent` profile. The Flash model's pool is much smaller than the 27B's;
  BF16 reduces recurrent-state memory at a different precision. The resulting
  capacity and quality measurements for this model are reported above.
- **`CPUSET=5-9,15-19`** — the sibling pinned SGLang to the 10 Cortex-X5 performance
  cores and kept it in every measured config. Exposed as `CPUSET` → docker
  `--cpuset-cpus` and pinned by default for the same reason; `CPUSET=` (empty) runs
  the container on all 20 cores. The full concurrent configuration was measured
  on this model; the CPU pinning's independent contribution remains inconclusive.

**SGLang-specific (not ported, no vLLM equivalent in v0.30):**

- `--max-mamba-cache-size` / `--max-running-requests` / `--cuda-graph-max-bs-decode` —
  SGLang sizes the mamba pool and the running-request cap as separate flags; vLLM sizes
  the GDN pool from the remaining memory and caps concurrency with `--max-num-seqs` alone.
- `--mamba-full-memory-ratio` and `--mamba-radix-cache-strategy` — SGLang scheduler
  internals with no vLLM flag.
- `--speculative-num-draft-tokens` — the sibling found draft=16 gives +28–41%
  single-stream but −10–15% aggregate, and that every draft token is verified so quality
  is unaffected. vLLM's MTP knob is `num_speculative_tokens` (`MTP=`), which is not the
  same dial; `MTP=3` is what we validated here (+7% decode, −1 tournament point), so the
  wider-window finding is recorded for context, not wired up.
- `--mem-fraction-static 0.85` — the sibling's settled value; our `GPU_MEM` already
  covers this and its own measurements (0.80 stable, 0.875 OOM) are documented above.

**Measurement caveats that also apply here** — the sibling's results document the traps it hit (boot-to-boot variance, greedy non-determinism, truncation reading as a quality regression); the [Benchmarking](#benchmarking) section restates them as this repo's hygiene rules.

## DFlash profile measurements

Measured on the GX10 (2026-10-28, PR #62) with the same target checkpoint and matched
serving settings, a 500k context window and `DRAFT_TOKENS=4`, comparing the default
MTP=2 draft head against the incoai DFlash2 draft model
([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)).
Rates below are medians; generation throughput includes request latency, and concurrent
rates are totals across all requests. No quality degradation was observed in our validation.

| Metric | MTP=2 | DFlash K=4 | Improvement |
| --- | ---: | ---: | ---: |
| Single-request generation | 45.2 tok/s | 57.0 tok/s | +26% |
| 4 concurrent requests: total throughput | 107.2 tok/s | 120.5 tok/s | +12% |
| 8 concurrent requests: total throughput | 167.0 tok/s | 176.1 tok/s | +5% |
| 100k-token prefill | 3,094 tok/s | 3,209 tok/s | +4% |

**Memory trade-off:** DFlash uses more GPU memory than native MTP. Allow extra headroom
for the drafter and serving caches: a memory budget that works with MTP may be
insufficient for DFlash at the same context length or concurrency. Check available memory
with `./flash doctor dflash` and size the serving memory budget for your workload.

## Benchmarking

`bench/` is a self-contained suite that measures this recipe the same way the
[Qwen3.8-27B-on-DGX-Spark](https://github.com/aphex3k/Qwen3.8-27B-on-DGX-Spark/blob/main/results/RESULTS.md)
recipe was measured on the same hardware, so the two can be compared head to head:

```bash
export GB10_BASE_URL=http://127.0.0.1:18300/v1   # wherever `./flash serve` listens
export GB10_MODEL=qwen3.8-flash-next

python3 bench/perf.py             # TTFT, single-stream decode, concurrency sweep, prefill curve
python3 bench/longctx.py          # concurrent long-context; unique content per stream
./scripts/run-humaneval.sh        # HumanEval pass@1, low reasoning effort
./scripts/run-humaneval.sh think  # xhigh reasoning effort
```

Everything is env-driven (`GB10_BASE_URL`, `GB10_API_KEY`, `GB10_MODEL` — see
`bench/common.py`) and the scripts are self-locating, so nothing is path-dependent.
HumanEval needs one-time setup on the box:
`python3 -m pip install --user pandas pyarrow huggingface_hub`; the dataset lands in
`data/humaneval`, results in `results/` (both gitignored), and candidates are executed
against their real unit tests in a `--network none` `python:3.12-slim` container.
The scripts select `reasoning_effort=low` or `xhigh` through the checkpoint's
template (`GB10_REASONING_OFF` / `GB10_REASONING_ON` override these values).
On the tested NVIDIA checkpoint, **`low` still enables thinking**; disabling it
requires `chat_template_kwargs={"enable_thinking": false}`, which the current
benchmark client does not send. The [PR #54 measurements above](#measured-concurrent-profile-pr-54)
used a separate client with thinking explicitly disabled and uncached prefill
verified from response usage. The bundled streaming client also drops
`finish_reason`, and `longctx.py` prints a placeholder zero for cached tokens;
these outputs do not verify truncation or an uncached prefill.

Measurement hygiene — each of these has produced a wrong number before:

- **Run on an idle box.** Any competing GPU work skews every figure.
- **Boot-to-boot single-stream variance is ~8%**, against <2% run-to-run within one
  instance. Size A/B deltas against that band; re-measure a small win on a fresh boot.
- **Greedy is not bitwise-stable across batch shapes** (see the deterministic top-k
  section); treat a pass@1 delta of ±2 HumanEval problems as noise.
- **Truncation reads as a quality regression.** A capped `max_tokens` truncates
  thinking runs; check `finish_reason == "length"` before quoting a score.
  `run-humaneval.sh` reports truncated problems separately from real failures.
- **Count `completion_tokens`, not stream events** — speculative decoding emits
  several tokens per event.
- **Long-context concurrency:** give every stream *unique* content, or the prefix cache
  deduplicates it and you measure cache hits instead of KV capacity. Prefill
  serialises, so read seconds-per-stream, not aggregate tok/s.
- **Concurrency sweep:** raise `SEQS` to at least the highest stream count you sweep,
  or excess requests queue silently and the aggregate flatlines (see above).
- **Health probes:** a model saturated on long prefills is healthy but too busy to
  answer a 5 s probe. If a supervisor watches the server, use ≥30 s timeouts or it will
  restart a healthy, busy server mid-job.
