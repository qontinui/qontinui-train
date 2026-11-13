"""ButtonYOLO against qontinui-train's own Poetry environment.

ButtonYOLO's default ``model_type`` is ``"yolov8"``, and its constructor always
loads a model, so the default path needs ``ultralytics`` installed. qontinui core
moved ultralytics into its optional ``[ml]`` extra, which a path dependency does
not inherit, so train declares it itself. This test is the guard that the env
built from this repo's ``poetry.lock`` can still construct the default model.
No CI job runs it yet (this repo has none): after a relock, run ``poetry sync`` and
then ``pytest`` by hand. ``tests/test_poetry_env.py`` checks the env against
``poetry.lock``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Runs in a child interpreter. Importing ultralytics and building a model changes
# process-wide state that monkeypatch cannot undo (OMP_NUM_THREADS and several
# other environment variables, numpy print options, the cv2 thread count, and the
# settings directory ultralytics binds at first import), so none of it may leak
# into the tests that run after this one.
_PROBE = """
import json, sys
from qontinui_train.button_detection.models.button_yolo import ButtonYOLO

detector = ButtonYOLO(model_type="yolov8", pretrained=False, device="cpu")
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(
        {
            "task": detector.model.task,
            "model_name": detector.model.model_name,
            "parameters": sum(p.numel() for p in detector.model.parameters()),
        },
        fh,
    )
"""


def test_yolov8_constructs_in_the_poetry_env(tmp_path: Path) -> None:
    """``pretrained=False`` builds YOLOv8n from the config bundled with ultralytics.

    No weights are downloaded: ultralytics builds YOLOv8n from its bundled
    ``yolov8.yaml`` at scale ``n``. The child keeps ultralytics' settings and
    torch's inductor cache under ``tmp_path``, writes no bytecode (``-B``), and
    runs offline, so it makes no network call. ``-E`` keeps a caller's
    ``PYTHONPATH`` from supplying an ultralytics the Poetry env lacks.
    When ultralytics is missing, the child fails with ButtonYOLO's "ultralytics
    package required for YOLOv8" ImportError.
    """
    result_file = tmp_path / "result.json"
    env = {
        **os.environ,
        "YOLO_CONFIG_DIR": str(tmp_path / "ultralytics-settings"),
        "YOLO_OFFLINE": "true",
        "TORCHINDUCTOR_CACHE_DIR": str(tmp_path / "torchinductor"),
    }
    proc = subprocess.run(
        [sys.executable, "-E", "-B", "-c", _PROBE, str(result_file)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr

    result = json.loads(result_file.read_text(encoding="utf-8"))
    assert result["task"] == "detect"
    assert result["model_name"] == "yolov8n.yaml"
    assert result["parameters"] > 0
