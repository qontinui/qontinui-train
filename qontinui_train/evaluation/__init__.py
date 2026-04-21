"""Evaluation scaffolding inside the qontinui_train package.

The primary eval harness (``grounding_eval.py``) lives one directory up,
at ``qontinui-train/evaluation/`` — that's the historical home. New eval
subsystems introduced for the VGA product surface live here instead,
under the proper importable package path so they can be imported as
``from qontinui_train.evaluation.external_app_splits import ...`` from
anywhere the package is installed.
"""
