# External-app held-out test split

Per-domain regression gate for the VGA retraining loop (plan §5
milestone (c), §13 v6-retrain-trigger).

## Purpose

Each production `target_process` (e.g. `notepad++.exe`, `obs64.exe`) has
its own held-out evaluation set built from **user-confirmed** entries in
the VGA correction log. The v6 (and later) grounding model is only
allowed to ship when it beats v5 by **≥ 5 percentage points Acc@center
on every domain's split**. Improving one domain while regressing another
is not acceptable and will fail the gate.

This is the synthetic-test-set analogue of production shadow evaluation:
`shadow_eval.py` (sibling module) compares model predictions on
live-logged samples; this module compares them on explicitly held-out,
confirmed ground truth.

## How samples are drawn

The loader in `loader.py` reads
`datasets/vga-corrections/corrections.jsonl` (or whatever directory is
passed in — same directory that `CorrectionLogger` writes to) and
includes an entry in the eval set when:

1. It is **not** private (or `include_private=True` is passed — not the
   default).
2. Either:
   - `source == "builder"`, meaning the user confirmed / corrected the
     bbox in the state-machine-builder UI; **or**
   - `test_reserved == true`, meaning the entry was explicitly marked
     as a held-out test sample.
3. The referenced image exists on disk.

Runtime-correction entries (`source == "runtime"`) are **excluded by
default** — those are retraining fuel, not regression gate input.

Samples are bucketed by `target_process`. The loader never mixes
domains.

## Per-domain ship gate

```
for domain in per_domain_splits:
    v5_acc  = evaluate(v5_model, domain_samples)["acc_center"]
    v6_acc  = evaluate(v6_model, domain_samples)["acc_center"]
    delta   = v6_acc - v5_acc
    assert delta >= 0.05, f"regression on {domain}: {delta:+.3f}"
```

If every `delta >= +0.05` and no domain regresses below its v5 baseline,
the supervisor-managed correction-loop daemon (plan §13 recommendation
E) is allowed to swap the `qontinui-grounding-v5` entry in
`llama-swap/config.yaml` for `qontinui-grounding-v6`. Otherwise v6 stays
available under its own model name for canary rollout.

## Running an eval against it

The benchmark is wired into `grounding_eval.py` through
`_BENCHMARK_PATHS` / `BENCHMARK_LOADERS` as
`external_app_<target_process>`. The loader is dynamic: each domain
present in the correction log produces one benchmark name.

```bash
# All domains at once — calls into load_per_domain_splits and iterates.
python -m qontinui_train.evaluation.shadow_eval \
    --pg-url postgres://runner@localhost/runner \
    --candidate-model qontinui-grounding-v6 \
    --baseline-model qontinui-grounding-v5

# Single domain via grounding_eval:
python -m evaluation.grounding_eval \
    --benchmark external_app_notepad++.exe \
    --baseline-model qontinui-grounding-v5 \
    --finetuned-model qontinui-grounding-v6 \
    --api-base http://localhost:5800/v1
```

(Path note: `grounding_eval.py` lives at `qontinui-train/evaluation/`,
not under the `qontinui_train/` package — the legacy module location
pre-dates the package move. The dynamic loader for
`external_app_<target_process>` is registered from the benchmarks
package at that legacy location.)

## Roadmap inside milestone (c)

- **Phase 1 (this module + exporter):** scaffold, selection rule,
  dynamic loader. No v6 yet. Running the eval against v5 establishes
  the baseline per-domain number.
- **Phase 2:** v6 training run on `vga-sft/` produced by
  `scripts/export_corrections_to_vlm_sft.py`. Compare against the
  baseline numbers recorded in phase 1.
- **Phase 3:** supervisor-managed daemon
  (`qontinui-finetune/scripts/correction_loop_daemon.py`) automates
  retrain + gate enforcement.
