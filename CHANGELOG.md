# Changelog

Notable changes to this recipe, newest first. The README keeps only the
quickstart and the pointers; the full story of every release lives here.
Performance and tournament figures are historical for the base they were
measured on unless explicitly labeled otherwise. Earlier updates (the preview
image and the v0.29/v0.30 bases) are in [docs/HISTORY.md](docs/HISTORY.md).

## 2026-10-28 — dflash/DFlash2 drafter support

The `dflash` profile replaces the MTP head with incoai's
[Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2),
a 0.6B block-diffusion drafter (one-time 1.2 GiB download, `./flash setup dflash`).
vLLM refuses this combination on this target out of the box — the five blockers
are catalogued in
[vllm#56088](https://github.com/vllm-project/vllm/issues/56088) — so **patch 23**
opens the speculative-method allowlist, mounts the target-side aux hidden-state
capture the drafter conditions on (completed-layer HC-mean taps, incoai's own
recipe), and backports the upstream QSA ring fix
[#54912](https://github.com/vllm-project/vllm/pull/54912). Thanks to
**[@roozich](https://github.com/roozich)** for the patch and
[PR #62](https://github.com/blazux/qwen3.8-Flash-DGX/pull/62).

## 2026-10-22 — official vLLM v0.32 base (issues #58, #59 and #60)

The recipe now builds on the official v0.32.0 image. Existing prepared hybrid
checkpoints are reusable. Rebuild with `./flash setup`, then recreate the server
with your usual profile and overrides. The checkpoint and profile defaults are
unchanged. See [port details and validation](docs/VLLM-032.md).

Blockwise-FP8 hybrid QSA keeps separate QKV and indexer projections to preserve
the existing checkpoint scale layout; other layouts retain upstream projection
fusion. QSA supports BF16 and FP8-e4m3 KV, including upstream layer skips; this
port does not enable NVFP4 KV. See the linked validation for the tested settings
and measurements. The v0.32 probes show comparable throughput, with no
established speedup. Performance and tournament figures in the README remain
historical unless explicitly labeled v0.32; the full tournament has not been
rerun for this release.

**Requested by** **[@neo42man](https://github.com/neo42man)**
([#58](https://github.com/blazux/qwen3.8-Flash-DGX/issues/58)), with an
independent patch audit by **[@rp-tn](https://github.com/rp-tn)**
([#59](https://github.com/blazux/qwen3.8-Flash-DGX/issues/59)). Preserves hybrid
QSA scale layout, mmap PLE, tool parsing and Responses fixes.

### Long-context MTP crash fix (issue #60)

New builds include patch 22 for the v0.32 M-RoPE cache sizing regression reported
by [@FinSnow](https://github.com/FinSnow) in
[#60](https://github.com/blazux/qwen3.8-Flash-DGX/issues/60). The MTP drafter's
cos/sin table could stop at **262,144 positions** while the server accepted
YaRN-scaled prompts up to 500k, allowing an out-of-bounds read and an EngineCore
crash during generation. Patch 22 extends the cache to cover the served target
window in both paths, preserving existing table rows and the YaRN frequency base.
It is adapted from [@wtogami](https://github.com/wtogami)'s
[vLLM #62409](https://github.com/vllm-project/vllm/pull/62409).

Existing v0.32 installations need an explicit rebuild and container recreation:

```bash
docker build -t qwen38-flash-dgx:v0.32 .
# Recreate the server with your usual profile and overrides.
```

`./flash setup` reuses an existing image with the current base label. For an
unpatched image, disabling MTP (`MTP=0`) or using the native-context profile are
the reported workarounds. See
[reproduction and validation](docs/VLLM-032.md#long-context-mtp-cache-fix-issue-60).

## 2026-10-19/20 — parser, tool-call and MTP fixes (issues #31, #55, #56)

| Date | Change |
| --- | --- |
| 2026-10-20 | **Schema-validated tool arguments:** patch 21 ports **[@Chaos2522](https://github.com/Chaos2522)**'s repair for accidental `arguments` wrappers to v0.31, preserving valid payloads and Chat compatibility ([#31](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6291191080)). [Behavior and validation](docs/PARSER-FIXES.md#schema-validated-tool-arguments-issue-31). |
| 2026-10-19 | **MTP + routed-expert output:** patch 20 clears auxiliary outputs only on the MTP draft config, fixing the inherited-config error. Thanks to **[@rp-tn](https://github.com/rp-tn)** for the diagnosis and patch ([#55](https://github.com/blazux/qwen3.8-Flash-DGX/issues/55)); [details and test scope](docs/PARSER-FIXES.md#routed-expert-output-with-mtp-issue-55). |
| 2026-10-19 | **Chat tool-name compatibility:** restore unoffered-name passthrough for Chat Completions while keeping Responses strict, following **[@FinSnow](https://github.com/FinSnow)**'s report ([#56](https://github.com/blazux/qwen3.8-Flash-DGX/issues/56)). [Behavior and validation](docs/SAFE-OUTPUT.md#chat-completions-compatibility-follow-up-issue-56). |
| 2026-10-16 | **Parser and Responses integrity:** patch 19 ports **[@Chaos2522](https://github.com/Chaos2522)**'s fixes for token identity, empty STOP and incomplete terminal events to v0.31 ([#31](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31)). [Details](docs/PARSER-FIXES.md#parser-and-responses-output-fixes-issue-31). |

Full behavior, validation numbers and reproduction instructions for all of
these are in [docs/PARSER-FIXES.md](docs/PARSER-FIXES.md).

## 2026-10-14 — concurrent profile and benchmarks (PR #54)

**[@aphex3k](https://github.com/aphex3k)**'s
[PR #54](https://github.com/blazux/qwen3.8-Flash-DGX/pull/54) adds CPU pinning,
BF16 recurrent-state control, `concurrent` and the `bench/` suite. Measured
aggregate throughput gains are **+21–94% at 8–32 clients** with the tested FP8
KV settings; [configuration, quality results and limits](docs/BENCHMARKS.md#measured-concurrent-profile-pr-54).

## 2026-10-09 — hybrid preparation and memory diagnostics

Support regular-file and hardlinked checkpoint indexes without modifying the
original snapshot ([#50](https://github.com/blazux/qwen3.8-Flash-DGX/issues/50));
add [optional startup-memory diagnostics](docs/VLLM-031.md#optional-diagnostic-images)
for automatic KV sizing ([#51](https://github.com/blazux/qwen3.8-Flash-DGX/issues/51)).

## Update 2026-10-08 — vLLM v0.31 migration

The v0.31 migration used the official v0.31.0 image. Existing prepared checkpoints can
be reused without conversion. Build and run:

```bash
./flash setup
./flash serve
./flash wait
./flash test
```

Tested on GX10 with the NVIDIA hybrid checkpoint, MTP=2 and FP8 KV caching:
the agentic tournament, prefix caching, sequential greedy determinism and
long-context retrieval at 499,000 input tokens. See the
[v0.31 validation results](docs/VLLM-031.md#gx10-validation-2026-10-07-to-2026-10-08)
for quality and performance measurements.

## Update 2026-09-28 — vLLM v0.30 became the stable base

- **One base image, one `Dockerfile`.** It builds the recipe on the vLLM v0.30.0 release
  (`qwen38-flash-dgx:v0.30`) and every profile uses it. v0.30 has run our own box since 2026-09-25:
  same tournament score as the v0.29 image before it, cold prefill 1.5–2× faster, ~3½-minute boots,
  and the fp8 KV cache for 1M-token context. The preview and v0.29 images, `Dockerfile.v0.29`, the
  `v0.29`/`v0.30` profiles and `PAD_M4` are gone. They are kept at the git tag
  [`multi-base-final`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/multi-base-final) if you need them.
- **Upgrading:** `git pull && ./flash setup && ./flash serve`. `flash` sees that your image is an older
  build and rebuilds it; the weights and the hybrid layout are reused as they are. If you ran
  `./flash serve v0.30`, it is now plain `./flash serve` (same recipe). By hand:
  `docker build -t qwen38-flash-dgx:v0.30 .` and `scripts/serve.sh` as before.
- **Earlier updates**, including the v0.30 vs v0.29 head-to-head, are in [docs/HISTORY.md](docs/HISTORY.md).

## Rebuilding for source-only fixes

For source-only fixes on an existing v0.32 installation, **rebuild the image**:
`./flash setup` skips an already-built image with the current base label.

```bash
docker build -t qwen38-flash-dgx:v0.32 .
# Recreate the container with your usual profile and overrides.
```
