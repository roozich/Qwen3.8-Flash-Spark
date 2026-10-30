# Troubleshooting

Setup and run problems that have actually happened here, with what fixed them.
Start with `./flash doctor [profile]` for anything: it checks the machine, memory,
docker, the port, the image, the checkpoint and the prepared layouts, and prints
what is missing.

## Downloading the checkpoint

**Download is slow.** `scripts/download-weights.sh` uses the Hugging Face Xet backend
(`XET=1`, the default). It used to be off because it stalled on some Spark setups, but the
Hub now refuses to serve files over 50 GB through the plain path at all: the NVIDIA
checkpoint's PLE table is one 50 GiB shard, and the error it prints ("install hf_xet") is
misleading, hf_xet is in the image. Xet is also much faster: `--max-workers` parallelises
across *files*, so a checkpoint that is a dozen large shards leaves most of a gigabit idle
over plain HTTPS. Measured by [@techfury90](https://github.com/techfury90) on a DGX Spark
on gigabit fibre, pulling 81 GB (we saw the same ~105 MB/s on ours):

| | rate | 81 GB takes |
| --- | --- | --- |
| plain HTTPS, 8 workers (default) | 14.7 MB/s (117 Mbit/s) | ~92 min |
| `XET=1` | **101 MB/s (809 Mbit/s)** | **13.4 min** |

`XET=0` falls back to plain HTTPS if Xet stalls for you.

**Xet run ends with `httpx.ReadTimeout` after the last file.** Seen once: the timeout hit
*after* the last file completed; every blob was intact, but the exit code was non-zero.
Re-run to confirm; the download is resumable, and a finished download re-checks in seconds.

**A file or shard is missing.** `./flash doctor` tells you if an essential file (tokenizer,
configs) or a shard named by the index is missing. The download is resumable: just run it
again.

## Preparing the hybrid layout

**vLLM dies with "Couldn't instantiate the backend tokenizer… sentencepiece".** That means
`prepare-hybrid.sh` ran on an *unfinished* download. The hybrid layout is a copy of the
snapshot as it is at that moment; prepared too early it lacked `tokenizer.json` (issue #17:
nothing is missing from the image). Since 2026-09-14 the script refuses an incomplete
snapshot and, run again on an existing layout, repairs it by adding whatever the snapshot
has gained; `serve.sh` and `flash` check the layout too and print that fix instead of
vLLM's message. Finish the download, then re-run `./flash setup`.

## Booting and memory

**The container dies during boot or the KV pool is tiny.** vLLM needs `GPU_MEM` ×
*currently free* memory to boot, not × 128 GB: anything else running on the box (another
model, a browser with big tabs) shrinks the pool. `./flash doctor` prints how much of the
pool is free right now. Close other GPU users or lower `GPU_MEM`.

**Don't co-locate another large model.** At `GPU_MEM=0.80` this recipe uses most of the
128 GB pool; an 8B embedding model next to it already starves the KV cache.

**DFlash boots but OOMs under load.** The `dflash` profile uses more GPU memory than the
native MTP head. A memory budget that works with MTP may be insufficient for DFlash at
the same context length or concurrency; check with `./flash doctor dflash` and size
`GPU_MEM` for the workload. Details: [DFlash measurements](BENCHMARKS.md#dflash-profile-measurements).

**1M context gets OOM-killed on a long prefill.** In bf16 a single 1M request needs
~26 GiB of KV; 500k with YaRN is the validated ceiling (800k booted but was OOM-killed on
a long prefill). For 1M use the `context-1m` profile (fp8 KV, `KV_DTYPE=fp8_e4m3`), which
costs some speed.

## While it runs

- `./flash status` shows the container state, KV pool, active patches, running requests
  and the prefix-cache hit rate; `./flash logs` follows the log
  ("Application startup complete" = ready).
- `tools/vllm_watch.py` is a live session viewer (see `tools/`).
- Slow first decode after boot is the page cache warming for the mmap'ed PLE table;
  it settles within the first requests.
