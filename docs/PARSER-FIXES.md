# Parser and tool-call fixes
Fixes shipped in the image for the Qwen tool-call parser, the Responses API output integrity and the MTP/routed-expert draft config, with their validation scope. Migrated from the README during the docs restructure; see also [docs/SAFE-OUTPUT.md](SAFE-OUTPUT.md) and the [changelog](../CHANGELOG.md).

## Quoted tool markers

The image includes a parser fix for literal or malformed `<tool_call>`
markers in Qwen reasoning and ordinary text. Previously, quoting that marker could
switch the parser into a tool preamble and discard subsequent text, including a
final answer after `</think>`. The parser now buffers the marker and preserves it as text when ordinary prose
follows, while recognizing a function-header prefix (`<function=`) as a tool call.
Valid calls and existing empty-wrapper/end-of-stream handling are retained.
The fix is enabled for the `qwen3` parser configuration, which is what
`--tool-call-parser qwen3_coder` (the flag `scripts/serve.sh` uses) and
`qwen3_xml` both run; derived parser configurations retain their existing behavior.

That fix covers a marker followed by ordinary prose. It does not cover a marker
followed by a well-formed function header — which is exactly what the model writes
when it *documents* the format, inside a ```` ```xml ```` block or while reasoning
about tool syntax. There the parser confirms the call, and everything after it is
consumed as tool-call syntax: with tools in the request you get a tool call the
model never meant to make, and without them the serving layer drops the call and
returns `content: null`, so the whole answer disappears. The visible output stops at
the fence opener, which is why this reads as "output dies on a backtick".

Patch 13 adds a second guard, on the same `qwen3` configuration (so it also covers
`qwen3_coder` and `qwen3_xml`):

- **Inside a fenced code block**, `<tool_call>` and `<function=` stay text and open
  no call. Fences follow CommonMark: a run of three or more backticks or tildes at
  the start of a line (at most three spaces of indent) opens one, and only a run of
  the same character at least as long, with nothing but whitespace after it, closes
  it — so a ```` ```xml ```` block nested inside a ````` ````md ````` block does not
  close the outer one. Fence state is per channel and is re-synced on use, so a
  fence left open in reasoning never carries into the answer, even when the call
  follows `</think>` with nothing in between.
- **A wrapper-less `<function=` header** opens a call only at the start of a line.
  Mid-line it is prose naming the marker. This is the `(CONTENT, FUNC_PREFIX)`
  fallback, which patch 12 does not guard at all.

Not covered, both deliberate:

- Illustrative `<tool_call>` XML written in reasoning *outside* a fence still opens
  a call. Suppressing that would mean dropping the reasoning → tool-call transition,
  which this model does use.
- A fence the model opens and never closes keeps the guard active for the rest of
  the turn, so a real tool call after it is returned as text instead of being
  parsed. That is the cost of deciding from the text alone;
  `test_unclosed_fence_suppresses_later_calls` asserts it so a change is deliberate.

The regression patches extend vLLM's Qwen parser tests. To run them from a vLLM v0.32.0
source checkout with its test dependencies installed (using absolute paths to
this repository's patch files):

```bash
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-preamble.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-preamble-tests.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-marker-guard.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-marker-guard-tests.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-safe-output.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-argument-schema.patch
.venv/bin/python -m pytest tests/parser/engine -q
```

Patch 13's own module, `tests/parser/engine/test_qwen3_literal_markers.py`, is 30
cases over three chunk sizes: 21 fail and 9 pass on patch 12 alone, all 30 pass with
patch 13. The 9 that pass either way are the no-regression guards — a real call
still parses, a real call after a closed fence still parses, and patch 12's own
inline-quoted-marker case is unchanged.

## Parser and Responses output fixes (issue #31)

New builds include [Chaos2522's output-integrity fixes](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6191209070),
ported to vLLM v0.31 as patch 19:

- Qwen final parsing retains original token IDs, preserving ordinary text that
  spells protocol markers instead of treating it as an atomic control token.
- An empty Responses STOP is reported as a failure instead of a successful empty
  answer. Streaming uses `response.failed`; nonstreaming returns an HTTP 500 error.
- A token-limited Responses generation ends with `response.incomplete`, including
  when the only generated content is hidden reasoning.

Chat Completions keeps incremental streaming for normal tool arguments. Responses buffers
converted XML arguments until the call closes so truncated calls do not emit
unfinished JSON. Chat Completions also keeps its original tool-name contract:
unoffered names are returned unchanged for the client to reject or recover from,
while known `functions.` / `tools.` aliases resolve to an offered name. Clients
must validate tool names before execution. Responses enforces the offered-name
list and reports invalid calls as failures. This API distinction avoids turning
a rejected Chat tool call into an empty successful turn (issue #56).
These fixes change parsing and error reporting; they do not prevent the model
from emitting EOS early.

Initial patch-19 validation on GX10: **67/67 targeted methods and 21/21 literal-byte fixtures**,
no additional failures in 4,782 parser-engine cases, and real-inference checks of
terminal events and tool round trips. The 55-scenario tournament was used as a
regression check; its score is not evidence of these bug fixes. See
[scope and validation](SAFE-OUTPUT.md) and the
[reproduction suite](../tests/issue31_safe_output/README.md).
The subsequent Chat compatibility fix adds **10 CPU regression methods**;
patch 20 has **4 CPU configuration checks**. These and the original 67 + 21
checks pass, with no new parser-engine failures.

To update an existing installation, rebuild with
`docker build -t qwen38-flash-dgx:v0.32 .`, then recreate the container with your
usual profile and overrides. `./flash setup` reuses an existing v0.32 image and
does not rebuild it for source-only fixes.

### Schema-validated tool arguments (issue #31)

Patch 21 fixes Qwen tool calls that accidentally wrap their parameters in a single
`arguments` property. For a tool requiring `cmd`, for example,
`{"arguments":"{\"cmd\":\"printf fixture\"}"}` becomes
`{"cmd":"printf fixture"}` only when the offered schema validates that repair.
Already valid payloads, including tools with a legitimate `arguments` parameter,
remain unchanged. Missing or ambiguous schemas do not trigger inferred repair.

The correction applies to Chat Completions and Responses. Responses reports
schema-invalid calls as failures before exposing an executable function-call item.
Chat retains its existing client recovery for invalid arguments and unknown names;
normal arguments stream incrementally, while a possible wrapper waits for closure.

**Thank you [@Chaos2522](https://github.com/Chaos2522) for the fix and regression
fixtures**, shared in [issue #31](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6291191080).
This is a targeted v0.31 port. CPU checks pass for **11/11 producer integration
tests, 10/10 Chat compatibility tests, 67/67 previous regression methods and
21/21 byte-fidelity cases**. The author's schema suite is **22/23**, with an
existing dotted-namespace limitation; the parser-engine suite remains
**4,781/4,782**, with its existing literal-marker test failure. No new failures
were introduced in these checks. These tests use synthetic engine frames;
no model-inference, speed or EOS improvement is claimed for patch 21.
See the [reproduction suite](../tests/issue31_schema_arguments/README.md) for details.
Existing installations need the image rebuild described above.

### Routed-expert output with MTP (issue #55)

Patch 20 fixes v0.31's draft-config copy when MTP is combined with
`--enable-return-routed-experts`: only the target keeps auxiliary-output capture;
the MTP drafter receives a fresh disabled auxiliary-output configuration. This
avoids the draft's `AuxOutput Connector only supports generate runners` error.
The target's other routed-expert compatibility requirements still apply.
**Thank you [@rp-tn](https://github.com/rp-tn) for the diagnosis and patch**, shared
in [issue #55](https://github.com/blazux/qwen3.8-Flash-DGX/issues/55) and integrated here.
The [CPU regression test](../tests/test_mtp_aux_output_cpu.py) checks the config copy
and vLLM's actual compatibility guard; it does not exercise GPU boot or delivery
of routed-expert output.
