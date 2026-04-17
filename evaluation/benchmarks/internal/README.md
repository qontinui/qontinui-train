# Internal grounding benchmark

Canned test set resolved by `grounding_eval.py --benchmark=internal`.

## Resolution

When `--benchmark=internal` is passed, `grounding_eval.py` resolves
`--test-jsonl` to:

```
qontinui-train/evaluation/benchmarks/internal/test.jsonl
```

(Found relative to the `grounding_eval.py` file's parent package — the path
is computed as `Path(__file__).parent / "benchmarks" / "internal" / "test.jsonl"`
so it works regardless of CWD.)

If that file is missing, `--benchmark=internal` errors out with a clear
message rather than silently falling back.

## Expected format

JSONL, one VLM-SFT sample per line. Each line is a JSON object with a
`"messages"` array matching the format produced by `qontinui-train`'s
`grounding_to_vlm.py`:

```jsonc
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "file:///abs/path/to/screenshot.png"},
        {"type": "text",  "text": "Click the primary Button labeled 'Save'"}
      ]
    },
    {
      "role": "assistant",
      "content": "<point>0.523 0.418</point>"
    }
  ]
}
```

Image paths in the test JSONL must resolve on the machine running
`grounding_eval.py` (so `file:///` paths relative to this repo, or absolute
container paths if running inside the eval-desktop image).

## Populating the benchmark

The canonical internal benchmark is a reserved slice of the synthetic grounding
pipeline's held-out `vlm_test.jsonl` (see
`proj_grounding_capture_pipeline.md` — the test set is pure-static records
only). To populate:

1. Run the capture → VLM-format pipeline per
   `qontinui-train/docs/grounding_capture.md`
2. Copy `dataset/vlm_sft/vlm_test.jsonl` to this directory as `test.jsonl`
3. Copy the referenced screenshot tree so the `file:///` URIs resolve

A `make benchmark-internal` target is planned but not yet wired —
until then, copy manually.

## Why this file isn't checked in

The test JSONL references ~MB-scale PNG screenshots. Keeping them out of git
avoids repo bloat; CI fetches them via the grounding-pipeline artifact store
before running eval.
