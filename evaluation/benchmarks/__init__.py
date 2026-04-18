"""External grounding benchmark loaders.

Each loader fetches a public grounding dataset from HuggingFace (or a mirror),
normalizes it into the VLM SFT sample format consumed by
:func:`grounding_eval.load_test_samples`, and caches a local shadow
``test.jsonl`` so subsequent runs are offline.

The normalized JSONL lines match the "messages" schema produced by
``qontinui_train.export.grounding_to_vlm`` — i.e. what `grounding_eval.py`
already understands. We do NOT emit ``GroundingRecord`` instances here,
because ``grounding_eval.py`` consumes VLM-formatted samples, not grounding
records. The ``GroundingRecord`` shape (see
``qontinui_train.export.grounding_record``) is the producer-side schema; the
loaders here land at the consumer-side (VLM) shape, which is derived from
GroundingRecord by ``grounding_to_vlm.py``.

Adding a new benchmark
----------------------
1. Create a new module here, e.g. ``my_benchmark.py``.
2. Expose a ``load(cache_dir: Path) -> Path`` function that:
   - downloads (or loads from HF cache) the dataset, pinned to a specific
     ``revision=`` when the dataset supports it;
   - normalizes each sample to the VLM-messages schema;
   - writes ``<cache_dir>/my_benchmark/test.jsonl`` if it doesn't exist;
   - returns the path to that file.
3. Register it in ``BENCHMARK_LOADERS`` below.
4. Append an entry to ``benchmarks/README.md``.

Bounding-box normalization
--------------------------
HF grounding datasets vary: some use ``[x1, y1, x2, y2]`` absolute pixels,
some use normalized ``[0,1]`` coords, some use ``[cx, cy, w, h]``.
Each loader picks one convention and documents it in its module docstring.
The VLM schema stores the *center point* in normalized ``[0,1]`` coords as
``<point>x y</point>`` in the assistant message, so every loader must
convert to that form before writing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .osworld_g import load as load_osworld_g
from .screenspot_pro import load as load_screenspot_pro
from .screenspot_v2 import load as load_screenspot_v2

# Registry of external benchmark loaders.
# Keys must not collide with ``_BENCHMARK_PATHS`` in ``grounding_eval.py``.
BENCHMARK_LOADERS: dict[str, Callable[[Path], Path]] = {
    "screenspot_v2": load_screenspot_v2,
    "screenspot_pro": load_screenspot_pro,
    "osworld_g": load_osworld_g,
}


__all__ = ["BENCHMARK_LOADERS"]
