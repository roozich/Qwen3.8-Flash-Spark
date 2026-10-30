# Upstream tracking

Status of every local patch that has (or should have) an upstream `vllm-project/vllm` counterpart.
Check this on **every new vLLM release**: if a tracked item is merged, verify the fix in the new
image, retire the local patch (keep it in tree, mark retired, add the build-time `grep -q` assert),
and record the migration in the version doc (`docs/VLLM-032.md` pattern).

Last checked: **2026-10-30** (GitHub API, live state).

## Open upstream items (local patch still needed)

| Local patch | Upstream item | Upstream state (2026-10-30) | Merge target check | Action when merged upstream |
|---|---|---|---|---|
| 5 — exact QSA top-k | issue [vllm#51782](https://github.com/vllm-project/vllm/issues/51782) → **superseded by [vllm#54521](https://github.com/vllm-project/vllm/issues/54521)** (open) | #51782 closed "completed" 10-26; the GB10 non-determinism itself is NOT fixed — follow #54521 | — | retire patches 5+8 together, keep `smoke-test.sh` determinism check as the gate |
| 8 — deterministic persistent_topk kernel | PR [vllm#55122](https://github.com/vllm-project/vllm/pull/55122) (@jschmied) | **open, unmerged** | grep the release tree for the deterministic kernel in `ops/*persistent_topk*` | drop the build-time nvcc compile; `VLLM_QSA_DET_TOPK` becomes a no-op |
| 4 — prefix-caching block_size | PR [vllm#54076](https://github.com/vllm-project/vllm/pull/54076) ("Use the Mamba cache group's block size for align-mode chunk splitting") | **open, unmerged** | grep release for the align-mode block-size use | retire patch 4, keep block-cache-hit smoke check |
| 22 — M-RoPE served window | issue [vllm#62383](https://github.com/vllm-project/vllm/issues/62383) + PR [vllm#62409](https://github.com/vllm-project/vllm/pull/62409) (@wtogami) | both **open** | diff `mrope.py` against upstream in the new image | retire patch 22 if the merged form covers YaRN base + cache rows (our adaptation reason) |
| 23 — dflash seam (allowlist + aux capture) | issue/recipe [vllm#56088](https://github.com/vllm-project/vllm/issues/56088) | **open** | grep Qwen4Exp spec allowlist in the new image | retire (a) allowlist + (b) aux capture parts independently |
| 23c — QSA ring widening (depths ≥ 5) | PR [vllm#54912](https://github.com/vllm-project/vllm/pull/54912) | **open, unmerged** | grep `common/qsa_cache.py` ring code | retire patch 23c part only |
| 19 — safe output / Responses events | none filed | — | — | candidate to file upstream (see `tests/issue31_safe_output/` fixtures, already upstream-shaped) |
| 12/13/21 — tool parser fixes | none filed | — | — | same: strong upstream-PR candidates, fixtures exist |
| 14–18 — weight-load speedups | none filed | — | — | performance-only; file if upstream shows interest in mmap-load paths |

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
