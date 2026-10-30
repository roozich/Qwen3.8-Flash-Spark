# Repository map

The full per-file map of the recipe: every patch, script, tool and test, and what it does.
Moved out of the README during the docs restructure; the front page is the [README](../README.md).

```
flash                             one-command front-end: doctor / setup / serve <profile> / wait / test / status …
profiles/*.env                    named recipes (default, speed, dflash, concurrent, context, context-1m, shared, published, native)
Dockerfile                        official vLLM v0.32.0 image + the patches below (qwen38-flash-dgx:v0.32).
                                     Native upstream now: 3, 7, 9, 11, 16 (docs/HISTORY.md, docs/VLLM-031.md)
src/vllm_ple_mmap.py              1. mmap PLE table (opaque op, breaks the CUDA-graph capture)  VLLM_PLE_MMAP=1
src/patch_mamba_block_size.py     4. prefix-caching block_size fix
src/patch_qsa_exact_topk.py       5. exact, deterministic QSA top-k                  VLLM_QSA_EXACT_TOPK=1
(Dockerfile patch 8)              8. deterministic persistent_topk kernel, built at docker build  VLLM_QSA_DET_TOPK=1
src/patch_qsadet.py                  wires that kernel into the QSA indexer
                                     from @jschmied's repo (pinned commit) — vllm#55122
src/patch_mtp_draft_vocab.py     10. reduced draft vocabulary for the MTP drafter          VLLM_MTP_DRAFT_VOCAB=<ids.npy>
src/draft_vocab_65536.npy            the default 65,536-id set (tools/build_draft_vocab.py rebuilds it)
src/vllm_fp8_hybrid_modelopt.py   6. NVFP4 experts + fp8 side layers dispatch        VLLM_FP8_HYBRID=1
                                     (patches the NVFP4 and the mixed-precision ModelOpt config classes)
src/patch_qsa_fp8_kv.py           7. retired local FP8 KV patch; v0.31 uses native upstream support
src/patch_moe_load_clone.py      14. clone mmap-backed expert weights before the H2D copy          VLLM_LOAD_CLONE=0 disables
                                     (main weight load 541 -> 150 s on a Spark; docs/HOW-IT-WORKS.md)
src/patches/qwen-tool-preamble*.patch / qwen-tool-marker-guard*.patch
                                12, 13. Qwen tool-marker fixes in the parser engine (+ their test modules)
src/patches/qwen-safe-output.patch 19. Qwen token identity, empty STOP failures and Responses terminal events
src/patches/qwen-mtp-aux-output.patch 20. keep target auxiliary outputs out of the MTP draft config
src/patches/qwen-tool-argument-schema.patch 21. schema-validated repair of accidental arguments wrappers
src/patches/qwen-mrope-cache-window.patch 22. cover the served target window in M-RoPE caches, including MTP
src/patch_dflash_qwen4exp.py     23. dflash/DFlash2 seam on Qwen4Exp: speculative-method allowlist +
                                     EAGLE-3-style target aux capture + QSA ring widening
                                     (vllm#56088; ring port of vllm#54912) - see profiles/dflash.env
src/test_dflash_patch_cpu.py        CPU unit test for patch 23 (anchors, idempotence, ring math)
src/patch_load_pread.py          15. pread checkpoint tensors <= 64 MiB instead of mmap views     VLLM_LOAD_PREAD=0 disables
                                     (never the PLE table; needs 14)
src/patch_moe_name_index.py      16. retired local expert-name index; v0.31 uses native upstream support
src/patch_embed_chunked_copy.py  17. embed_tokens / lm_head copied to the GPU in 64 MiB pieces     VLLM_LOAD_EMBED_CHUNK=0 disables
                                     15-17 together: weight load 150 + 32 s -> 35 + 12 s (docs/HOW-IT-WORKS.md)
src/patch_mtp_name_prefilter.py  18. MTP drafter skips non-MTP tensors before reading (12 -> 1.2 s) VLLM_MTP_NAME_PREFILTER=0 disables
src/test_ple_mmap_cpu.py          CPU unit test for the gather (no GPU needed)
src/test_qsa_exact_topk_cpu.py    CPU unit test for the exact top-k (no GPU needed)
src/test_moe_name_index_cpu.py    CPU unit test: patch 16 visits exactly the entries of the original loop
src/test_load_patches_cpu.py      CPU check of patches 15 and 17 against real checkpoint files
src/test_mtp_prefilter_cpu.py     CPU check of patch 18: the drafter's exact tensor set, on a real snapshot
tests/test_fp8_kv_read.py         GPU check of patch 7: the fp8 read path matches bf16 bit for bit
tests/issue31_safe_output/        patch-19 regression fixtures, Chat compatibility checks and client reproduction
tests/issue31_schema_arguments/   patch-21 schema and producer regressions, provenance and installed-source hashes
tests/test_mtp_aux_output_cpu.py  patch-20 draft-config isolation and auxiliary-output compatibility checks
tests/test_prepare_hybrid.py     synthetic checkpoint/index preparation regressions (issue #50)
tests/test_memory_profile_logging.py CPU checks for the optional startup-memory diagnostic hook
tools/fp8_convert.py              side-layer bf16 -> blockwise fp8 (by @Saren-Arterius)
tools/bench_moe_load.py           per-expert H2D copy micro-benchmark behind patch 14 (needs a free GPU)
tools/profile_boot.sh             py-spy every process of a boot in 10 s slices (needs SYS_PTRACE, see header)
tools/pyspy_slices.py             summarize those slices by process / thread / frame
tools/Dockerfile.memory-diagnostics optional logging-only image for automatic KV sizing (issue #51)
tools/patch_memory_profile_logging.py startup memory snapshots without allocation changes
scripts/download-weights.sh       MODEL (default nvidia/Qwen3.8-Flash-Next-NVFP4), EXCLUDE, MAX_WORKERS, XET
scripts/download-draft.sh         one-time: the DFlash2 draft model (DRAFT_MODEL, default incoai/Qwen3.8-Flash-Next-DFlash2, ~1.2 GiB) for the dflash profile
scripts/prepare-hybrid.sh         one-time: build the -fp8hybrid snapshot
scripts/prepare-mtp-graft.sh      one-time: graft the NVFP4 MTP draft experts onto it (MODE=hybrid-mtp, RadixArk only)
tools/vllm_watch.py               live per-session view of prompts / reasoning / outputs / stats (needs LOG_REQUESTS=1; @0x3dlux)
scripts/serve.sh                  MODE=nvfp4|hybrid|hybrid-mtp, SERVED_MODEL_NAME, PREFIX_CACHE, DET_TOPK, DRAFT_VOCAB, DRAFT_MODEL, DRAFT_TOKENS, MADVISE, EXACT_TOPK, KV_DTYPE, YARN, ...
scripts/smoke-test.sh             health, coherence, prefix-cache hit, determinism, tok/s
scripts/greedy-probe.sh           greedy probe set; diff two arms to gate a draft/checkpoint swap
scripts/run-humaneval.sh          HumanEval end-to-end: generate -> sandboxed execution -> report
bench/                            perf.py (TTFT/decode/concurrency/prefill), longctx.py, humaneval/ — the benchmark suite
docs/HOW-IT-WORKS.md              how each patch works, with the measurements
docs/VLLM-031.md                  v0.31 migration, serving validation and automatic KV sizing diagnostics
docs/SAFE-OUTPUT.md               parser/Responses fixes, Chat compatibility and validation scope
docs/HISTORY.md                   the earlier updates and the preview / v0.29 bases
docs/BENCHMARKS.md                throughput, concurrency and benchmarking methodology
docs/CONFIGURATION.md             runtime options, checkpoint layouts, tuning reference
docs/PARSER-FIXES.md              parser/tool-call/Responses fixes and validation scope
docs/UPSTREAM-TRACKING.md         live status of upstream vLLM items per patch; check on every release
docs/REPO-MAP.md                  this map
CHANGELOG.md                      release notes, newest first
```
