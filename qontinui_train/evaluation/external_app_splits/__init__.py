"""Per-domain held-out test splits for the VGA external-app regression gate.

See ``README.md`` in this directory for the high-level design. The public
surface is:

- :func:`loader.load_per_domain_splits` — read the VGA correction log and
  produce one list of VLM-SFT-shaped samples per ``target_process``.
"""

from .loader import EvalSample, load_per_domain_splits

__all__ = ["EvalSample", "load_per_domain_splits"]
