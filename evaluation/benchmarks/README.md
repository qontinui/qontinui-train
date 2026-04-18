# Grounding benchmark loaders

Index of external grounding benchmarks wired into
`qontinui-train/evaluation/grounding_eval.py`.

## Benchmarks

| Name             | HF dataset                      | Revision pin env var                  |
|------------------|---------------------------------|---------------------------------------|
| `screenspot_v2`  | `rootsautomation/ScreenSpot`    | `QONTINUI_SCREENSPOT_V2_REV`          |
| `screenspot_pro` | `likaixin/ScreenSpot-Pro`       | `QONTINUI_SCREENSPOT_PRO_REV`         |
| `osworld_g`      | `xlangai/OSWorld-G`             | `QONTINUI_OSWORLD_G_REV`              |
| `internal`       | local `benchmarks/internal/test.jsonl` | n/a (repo-relative)            |

Each loader (except `internal`) lives in its own module here and exposes a
`load(cache_dir: Path) -> Path` function that:

1. Returns the cached `<cache_dir>/<name>/test.jsonl` if it already exists.
2. Otherwise fetches the HF dataset (pinned revision when set),
   normalizes each row to a VLM-messages sample matching
   `grounding_to_vlm.py`'s output, persists images under
   `<cache_dir>/<name>/images/`, and writes the jsonl.

## Cache layout

```
<cache_dir>/
  <benchmark_name>/
    test.jsonl           # VLM-messages samples
    images/
      <benchmark>_NNNNNN.png
```

Default `<cache_dir>` is `qontinui-train/.benchmark-cache/` (controlled by
`grounding_eval.py --benchmark-cache-dir`).

## Bbox normalization

HF grounding datasets disagree on bbox layout. Each loader documents its
assumption in its module docstring. All loaders emit the same
normalized center `<point>cx cy</point>` in `[0, 1]^2`, so
`grounding_eval.py`'s Acc@center / MDE math is benchmark-agnostic.

## Adding a new benchmark

1. Add a module `my_benchmark.py` with a `load(cache_dir)` function.
2. Register it in `__init__.py:BENCHMARK_LOADERS`.
3. Append a row to the table above.

## Published-number reproduction check

`reproduction_check.py` holds a small table of published UI-TARS-1.5-7B
Acc@center numbers for these benchmarks. When the eval harness runs
against a benchmark in that table, it logs a warning if the measured
number deviates by more than 2 percentage points. The source for the
table is documented in that module.
