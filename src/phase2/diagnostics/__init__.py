"""Phase 2 diagnostics — scientific audit of the IPF→δQP framework.

Modules
-------
* ``d1_true_map``: COCO-style mAP re-evaluator (replaces the legacy
  ``precision × recall`` computed in ``phase2/evaluation/task_accuracy.py``).
* ``d2_statistical_significance``: paired Wilcoxon, Cohen's d, and bootstrap
  CI per (sequence, QP) cell. Consumes per-frame counts from D1.
* ``d3_phi_distribution``: spatial entropy and percentile spread of the
  occlusion-saliency maps; quantifies how much "headroom" the IPF concept
  has on each sequence.

See ``phase2/docs/PROJECT_AUDIT.md`` for the full motivation and decision
tree.
"""
