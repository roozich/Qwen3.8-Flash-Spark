# Upstream tracking

Status of every local patch that has (or should have) an upstream `vllm-project/vllm` counterpart.
Check this on **every new vLLM release**: if a tracked item is merged, verify the fix in the new
image, retire the local patch (keep it in tree, mark retired, add the build-time `grep -q` assert),
and record the migration in the version doc (`docs/VLLM-032.md` pattern).

Last checked: **2026-10-30** (GitHub API, live state).

## Open upstream items (local patch still needed)

| Local patch | Upstream item | Upstream state (2026-10-30) | Merge target check | Action when merged upstream |
|---|---|---|---|---|
| 1 — PLE mmap gather | none filed (by design) | — | grep the release for a native equivalent of disk-backed PLE | intentionally local: a speedup over vLLM's own `VLLM_PLE_CPU_OFFLOAD=1`; native is the fallback, not a retirement target |
| 2 — GB10 FLA fixes | different upstream: `fla-org/flash-linear-attention` (vendored by vLLM) | see the patch-2 subsection below | grep the vendored copy (subsection) | retire when the **vendored** copy carries both fixes |
| 4 — prefix-caching block_size | PR [vllm#54076](https://github.com/vllm-project/vllm/pull/54076) ("Use the Mamba cache group's block size for align-mode chunk splitting") | **open, unmerged** | grep release for the align-mode block-size use | retire patch 4, keep block-cache-hit smoke check |
| 5 — exact QSA top-k | issue [vllm#51782](https://github.com/vllm-project/vllm/issues/51782) → **superseded by [vllm#54521](https://github.com/vllm-project/vllm/issues/54521)** (open) | #51782 closed "completed" 10-26; the GB10 non-determinism itself is NOT fixed — follow #54521 | — | retire patches 5+8 together, keep `smoke-test.sh` determinism check as the gate |
| 6 — NVFP4 + fp8 hybrid dispatch | none filed | — | — | local-only: it teaches ModelOpt configs about *our* prepared `-fp8hybrid` checkpoint, not a stock vLLM path |
| 8 — deterministic persistent_topk kernel | PR [vllm#55122](https://github.com/vllm-project/vllm/pull/55122) (@jschmied) | **open, unmerged** | grep the release tree for the deterministic kernel in `ops/*persistent_topk*` | drop the build-time nvcc compile; `VLLM_QSA_DET_TOPK` becomes a no-op |
| 12–21 — parser + output fixes (12/13 marker engine, 19 safe output, 20 MTP aux isolation, 21 schema repair) | none filed | — | — | the strongest upstream-PR candidates; `tests/issue31_*` fixtures are already upstream-shaped |
| 22 — M-RoPE served window | issue [vllm#62383](https://github.com/vllm-project/vllm/issues/62383) + PR [vllm#62409](https://github.com/vllm-project/vllm/pull/62409) (@wtogami) | both **open** | diff `mrope.py` against upstream in the new image | retire patch 22 if the merged form covers YaRN base + cache rows (our adaptation reason) |
| 23 — dflash seam (allowlist + aux capture) | issue/recipe [vllm#56088](https://github.com/vllm-project/vllm/issues/56088) | **open** | grep Qwen4Exp spec allowlist in the new image | retire (a) allowlist + (b) aux capture parts independently |
| 23c — QSA ring widening (depths ≥ 5) | PR [vllm#54912](https://github.com/vllm-project/vllm/pull/54912) | **open, unmerged** | grep `common/qsa_cache.py` ring code | retire patch 23c part only |

Local-only, no upstream ambition: **14–18** (weight-load speedups — performance-only, file only if
upstream shows interest in mmap-load paths).

### Patch 2 — the FLA fixes live in a different upstream (`fla-org/flash-linear-attention`)

vLLM ships flash-linear-attention vendored under `vllm/third_party/flash_linear_attention/`,
so patch 2 retires only when the **vendored copy** carries both fixes. Checked 2026-10-30 against
vLLM `main` and fla `main`:

| FLA fix (Dockerfile step 2) | Upstream status | Vendored copy (vLLM main, 2026-10-30) |
|---|---|---|
| `num_warps` pin (Blackwell `tl.dot` race) | fla issue [#953](https://github.com/fla-org/flash-linear-attention/issues/953) **closed as done 2026-06-20**; fla main now uses a `GATED_DELTA_RULE_FWD_H_NUM_WARPS` constant | **still `for num_warps in [2, 4]`** — vLLM has not re-vendored the fix |
| shared-memory gate `DEFAULT = 102400` | **not fixed upstream** — fla main still has `DEFAULT = 102400` with no sm_121/GB10 class (only AD102 = 101376), so GB10 (99 KiB) still falls off the big-tile path; no issue filed | **still `DEFAULT = 102400`** |

Retirement check per new release (both must pass before removing the sed):
`grep -q GATED_DELTA_RULE_FWD_H_NUM_WARPS <dist>/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py`
and the vendored gate no longer asking more than 101376 bytes on sm_121.
Action item: consider filing the GB10 shmem-gate issue upstream at fla-org (the constant, the
sm_121 = 101376 evidence, and the Qwen3.5-122B Spark precedent are all in `docs/HOW-IT-WORKS.md`).

## Already retired (reference: how retirement looks)

| Local patch | Merged upstream as | Version that shipped it |
|---|---|---|
| 3 — Mamba state-copy race | [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) | v0.30 |
| 9 — fp8 GEMM M%4 | [vllm#52775](https://github.com/vllm-project/vllm/pull/52775) | v0.30 |
| 11 — block-FP8 MTP experts | [vllm#55513](https://github.com/vllm-project/vllm/pull/55513) | v0.30 |
| 7 — fp8 KV cache | [vllm#55557](https://github.com/vllm-project/vllm/pull/55557) | v0.31 |
| 16 — expert-name index | [vllm#58720](https://github.com/vllm-project/vllm/pull/58720) | v0.31 |

## How to re-check in one minute

```bash
# state of each tracked item (state / merged):
for n in 54521 54076 62383 62409 56088 54912 55122; do
  echo -n "$n: "; curl -s "https://api.github.com/repos/vllm-project/vllm/issues/$n" \
    | grep -Eo '"state": "[a-z]+"|"merged_at": "[^"]*"' | tr '\n' ' '; echo
done
```

Or in VS Code: run `/track-vllm-upstream` — it queries the same items, compares against this table,
and updates the "Last checked" row plus any retired patch it finds.
