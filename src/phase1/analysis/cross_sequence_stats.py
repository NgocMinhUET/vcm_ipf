"""Cross-sequence statistical analysis for multi-video validation.

Aggregates comparison results from multiple sequences and computes:
    1. Per-sequence metric tables
    2. Cross-sequence mean +/- std
    3. Paired t-test (IPF v2 vs each baseline)
    4. Effect size (Cohen's d)
    5. 95% confidence intervals via bootstrap
    6. Win/loss/tie summary

This module fulfills the statistical protocol defined in
04_EXPERIMENT_MATRIX.md section E.

Usage (CLI):
    python -m phase1.analysis.cross_sequence_stats \\
        --output-dir ~/ipf_outputs \\
        --run-prefix multi_seq_

Usage (Python):
    from phase1.analysis.cross_sequence_stats import CrossSequenceAnalyzer
    analyzer = CrossSequenceAnalyzer(output_dir, run_prefix="multi_seq_")
    report = analyzer.analyze()
"""

from __future__ import annotations

import json
import math
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


METRIC_NAMES = [
    "qp_std_of_means",
    "mean_frame_to_frame_delta",
    "max_frame_to_frame_delta",
    "per_ctu_temporal_std_mean",
    "temporal_jitter_index",
    "spatial_smoothness_mean",
]

METRIC_DISPLAY = {
    "qp_std_of_means": "QP Std",
    "mean_frame_to_frame_delta": "Dt Mean",
    "max_frame_to_frame_delta": "Dt Max",
    "per_ctu_temporal_std_mean": "CTU sigma_t",
    "temporal_jitter_index": "Jitter",
    "spatial_smoothness_mean": "Spatial Smooth",
}

LOWER_IS_BETTER = {
    "qp_std_of_means": True,
    "mean_frame_to_frame_delta": True,
    "max_frame_to_frame_delta": True,
    "per_ctu_temporal_std_mean": True,
    "temporal_jitter_index": True,
    "spatial_smoothness_mean": True,
}


@dataclass
class PairwiseTest:
    """Result of a pairwise statistical test."""
    method_a: str
    method_b: str
    metric: str
    mean_a: float
    mean_b: float
    diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    n_sequences: int
    significant: bool


@dataclass
class MethodSummary:
    """Aggregated metrics for one method across sequences."""
    method_id: str
    n_sequences: int
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    wins: int = 0
    losses: int = 0
    ties: int = 0


@dataclass
class CrossSequenceReport:
    """Complete cross-sequence analysis report."""
    sequences: List[str]
    methods: List[str]
    per_sequence: Dict[str, Dict[str, Dict[str, float]]]
    summaries: Dict[str, MethodSummary]
    pairwise_tests: List[PairwiseTest]
    analysis_mode: str


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size for paired samples."""
    diff = a - b
    d_mean = np.mean(diff)
    d_std = np.std(diff, ddof=1)
    if d_std < 1e-12:
        return 0.0
    return float(d_mean / d_std)


def _bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean(a - b)."""
    rng = np.random.default_rng(seed)
    diff = a - b
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(diff[idx])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lo, hi


def _paired_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Paired t-test returning (t_stat, p_value)."""
    if HAS_SCIPY:
        result = scipy_stats.ttest_rel(a, b)
        return float(result.statistic), float(result.pvalue)
    diff = a - b
    n = len(diff)
    d_mean = np.mean(diff)
    d_std = np.std(diff, ddof=1)
    if d_std < 1e-12 or n < 2:
        return 0.0, 1.0
    t_stat = d_mean / (d_std / math.sqrt(n))
    df = n - 1
    p_value = 2 * (1 - _t_cdf(abs(t_stat), df))
    return float(t_stat), float(p_value)


def _t_cdf(t: float, df: int) -> float:
    """Approximate Student's t CDF (fallback when scipy unavailable)."""
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5
    beta_inc = _regularized_beta(x, a, b)
    return 1.0 - 0.5 * beta_inc


def _regularized_beta(x: float, a: float, b: float, n_iter: int = 200) -> float:
    """Regularized incomplete beta function via continued fraction."""
    if x < 0 or x > 1:
        return 0.0
    if x == 0 or x == 1:
        return x
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1 - x)
    ) / a
    f, c, d = 1.0, 1.0, 0.0
    for i in range(n_iter):
        m = i // 2
        if i == 0:
            numerator = 1.0
        elif i % 2 == 0:
            numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10:
            break
    return front * (f - 1.0)


class CrossSequenceAnalyzer:
    """Analyzes comparison results across multiple video sequences."""

    def __init__(
        self,
        output_dir: str,
        run_prefix: str = "multi_seq_",
        analysis_mode: str = "steady",
    ):
        self.output_dir = Path(output_dir).expanduser()
        self.run_prefix = run_prefix
        self.analysis_mode = analysis_mode

    def _discover_runs(self) -> Dict[str, Path]:
        """Find all run directories matching the prefix."""
        runs = {}
        if not self.output_dir.exists():
            return runs
        for d in sorted(self.output_dir.iterdir()):
            if d.is_dir() and d.name.startswith(self.run_prefix):
                seq_name = d.name[len(self.run_prefix):]
                summary_file = d / "comparison_summary.json"
                if summary_file.exists():
                    runs[seq_name] = d
        return runs

    def _load_summary(self, run_dir: Path) -> dict:
        """Load comparison_summary.json from a run directory."""
        path = run_dir / "comparison_summary.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def analyze(self) -> CrossSequenceReport:
        """Run the full cross-sequence analysis."""
        runs = self._discover_runs()
        if not runs:
            raise FileNotFoundError(
                f"No runs found with prefix '{self.run_prefix}' in {self.output_dir}"
            )

        sequences = sorted(runs.keys())
        key = "methods_steady" if self.analysis_mode == "steady" else "methods_full"

        all_methods = set()
        per_sequence: Dict[str, Dict[str, Dict[str, float]]] = {}

        for seq_name, run_dir in runs.items():
            summary = self._load_summary(run_dir)
            methods_data = summary.get(key, summary.get("methods_full", {}))
            per_sequence[seq_name] = {}
            for mid, metrics in methods_data.items():
                all_methods.add(mid)
                per_sequence[seq_name][mid] = {
                    m: metrics.get(m, 0.0) for m in METRIC_NAMES
                }

        methods = sorted(all_methods)

        summaries = self._compute_summaries(sequences, methods, per_sequence)
        pairwise = self._run_pairwise_tests(sequences, methods, per_sequence)
        self._compute_win_counts(summaries, pairwise)

        report = CrossSequenceReport(
            sequences=sequences,
            methods=methods,
            per_sequence=per_sequence,
            summaries=summaries,
            pairwise_tests=pairwise,
            analysis_mode=self.analysis_mode,
        )

        self._write_reports(report)
        return report

    def _compute_summaries(
        self,
        sequences: List[str],
        methods: List[str],
        per_sequence: Dict[str, Dict[str, Dict[str, float]]],
    ) -> Dict[str, MethodSummary]:
        """Compute per-method cross-sequence statistics."""
        summaries = {}
        for mid in methods:
            values: Dict[str, List[float]] = {m: [] for m in METRIC_NAMES}
            for seq in sequences:
                if mid in per_sequence.get(seq, {}):
                    for metric in METRIC_NAMES:
                        values[metric].append(per_sequence[seq][mid].get(metric, 0.0))

            n_seq = len(values[METRIC_NAMES[0]])
            metrics_agg = {}
            for metric in METRIC_NAMES:
                arr = np.array(values[metric])
                metrics_agg[metric] = {
                    "mean": float(np.mean(arr)) if len(arr) > 0 else 0.0,
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "median": float(np.median(arr)) if len(arr) > 0 else 0.0,
                    "min": float(np.min(arr)) if len(arr) > 0 else 0.0,
                    "max": float(np.max(arr)) if len(arr) > 0 else 0.0,
                }

            summaries[mid] = MethodSummary(
                method_id=mid,
                n_sequences=n_seq,
                metrics=metrics_agg,
            )
        return summaries

    def _run_pairwise_tests(
        self,
        sequences: List[str],
        methods: List[str],
        per_sequence: Dict[str, Dict[str, Dict[str, float]]],
    ) -> List[PairwiseTest]:
        """Run pairwise statistical tests: M4 vs each baseline."""
        if "M4" not in methods:
            return []

        baselines = [m for m in methods if m != "M4" and not m.startswith("A")]
        tests = []

        for baseline_id in baselines:
            for metric in METRIC_NAMES:
                vals_m4 = []
                vals_bl = []
                for seq in sequences:
                    m4_data = per_sequence.get(seq, {}).get("M4", {})
                    bl_data = per_sequence.get(seq, {}).get(baseline_id, {})
                    if m4_data and bl_data:
                        vals_m4.append(m4_data.get(metric, 0.0))
                        vals_bl.append(bl_data.get(metric, 0.0))

                if len(vals_m4) < 2:
                    continue

                a = np.array(vals_m4)
                b = np.array(vals_bl)

                t_stat, p_value = _paired_t_test(a, b)
                d = _cohens_d(a, b)
                ci_lo, ci_hi = _bootstrap_ci(a, b)

                tests.append(PairwiseTest(
                    method_a="M4",
                    method_b=baseline_id,
                    metric=metric,
                    mean_a=float(np.mean(a)),
                    mean_b=float(np.mean(b)),
                    diff=float(np.mean(a - b)),
                    t_stat=t_stat,
                    p_value=p_value,
                    cohens_d=d,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    n_sequences=len(vals_m4),
                    significant=p_value < 0.05,
                ))

        return tests

    def _compute_win_counts(
        self,
        summaries: Dict[str, MethodSummary],
        tests: List[PairwiseTest],
    ) -> None:
        """Count wins/losses/ties for M4 vs each baseline."""
        if "M4" not in summaries:
            return

        for test in tests:
            lower_better = LOWER_IS_BETTER.get(test.metric, True)
            if test.significant:
                m4_better = test.diff < 0 if lower_better else test.diff > 0
                if m4_better:
                    summaries["M4"].wins += 1
                    if test.method_b in summaries:
                        summaries[test.method_b].losses += 1
                else:
                    summaries["M4"].losses += 1
                    if test.method_b in summaries:
                        summaries[test.method_b].wins += 1
            else:
                summaries["M4"].ties += 1
                if test.method_b in summaries:
                    summaries[test.method_b].ties += 1

    def _write_reports(self, report: CrossSequenceReport) -> None:
        """Write all analysis outputs."""
        report_dir = self.output_dir / "cross_sequence_analysis"
        report_dir.mkdir(parents=True, exist_ok=True)

        self._write_text_report(report, report_dir / "cross_sequence_report.txt")
        self._write_json_report(report, report_dir / "cross_sequence_report.json")
        self._write_per_sequence_csv(report, report_dir / "per_sequence_metrics.csv")
        self._write_pairwise_csv(report, report_dir / "pairwise_tests.csv")

    def _write_text_report(self, report: CrossSequenceReport, path: Path) -> None:
        """Write human-readable analysis report."""
        lines = []
        lines.append("=" * 120)
        lines.append("CROSS-SEQUENCE STATISTICAL ANALYSIS REPORT")
        lines.append(f"Mode: {report.analysis_mode} | Sequences: {len(report.sequences)}")
        lines.append("=" * 120)
        lines.append("")

        lines.append("SEQUENCES ANALYZED:")
        for seq in report.sequences:
            lines.append(f"  - {seq}")
        lines.append("")

        lines.append("-" * 120)
        lines.append("1. CROSS-SEQUENCE SUMMARY (mean +/- std)")
        lines.append("-" * 120)

        header = f"{'Method':<20}"
        for metric in METRIC_NAMES:
            header += f" {METRIC_DISPLAY[metric]:>18}"
        lines.append(header)
        lines.append("-" * 120)

        display_order = ["M0", "M1", "M4", "M5", "M6", "M7", "M8"]
        ablations = sorted([m for m in report.methods if m.startswith("A")])

        for mid in display_order + ablations:
            if mid not in report.summaries:
                continue
            s = report.summaries[mid]
            name = f"{mid} (IPF-v2)" if mid == "M4" else mid
            row = f"{name:<20}"
            for metric in METRIC_NAMES:
                m_data = s.metrics.get(metric, {})
                mean = m_data.get("mean", 0.0)
                std = m_data.get("std", 0.0)
                row += f" {mean:>8.3f}+/-{std:<7.3f}"
            lines.append(row)

        lines.append("")
        lines.append("-" * 120)
        lines.append("2. PAIRWISE TESTS: M4 (IPF v2) vs BASELINES")
        lines.append("-" * 120)
        lines.append(
            f"{'Baseline':<10} {'Metric':<18} {'M4 mean':>9} {'BL mean':>9} "
            f"{'Diff':>9} {'t-stat':>8} {'p-value':>9} {'Cohen d':>9} "
            f"{'95% CI':>20} {'Sig?':>5}"
        )
        lines.append("-" * 120)

        for t in report.pairwise_tests:
            sig_mark = "***" if t.p_value < 0.001 else (
                "**" if t.p_value < 0.01 else (
                    "*" if t.p_value < 0.05 else "ns"
                )
            )
            ci_str = f"[{t.ci_lower:+.4f}, {t.ci_upper:+.4f}]"
            lines.append(
                f"{t.method_b:<10} {METRIC_DISPLAY.get(t.metric, t.metric):<18} "
                f"{t.mean_a:>9.4f} {t.mean_b:>9.4f} "
                f"{t.diff:>+9.4f} {t.t_stat:>8.3f} {t.p_value:>9.4f} {t.cohens_d:>+9.3f} "
                f"{ci_str:>20} {sig_mark:>5}"
            )

        lines.append("")
        lines.append("-" * 120)
        lines.append("3. WIN/LOSS/TIE SUMMARY (M4 vs baselines, p<0.05)")
        lines.append("-" * 120)

        if "M4" in report.summaries:
            s = report.summaries["M4"]
            total = s.wins + s.losses + s.ties
            lines.append(f"  M4 Wins:   {s.wins}/{total}")
            lines.append(f"  M4 Losses: {s.losses}/{total}")
            lines.append(f"  Ties:      {s.ties}/{total}")

        lines.append("")
        lines.append("-" * 120)
        lines.append("4. PER-SEQUENCE BREAKDOWN")
        lines.append("-" * 120)

        for seq in report.sequences:
            lines.append(f"\n  Sequence: {seq}")
            seq_header = f"    {'Method':<20}"
            for metric in METRIC_NAMES:
                seq_header += f" {METRIC_DISPLAY[metric]:>12}"
            lines.append(seq_header)

            for mid in display_order + ablations:
                if mid not in report.per_sequence.get(seq, {}):
                    continue
                data = report.per_sequence[seq][mid]
                row = f"    {mid:<20}"
                for metric in METRIC_NAMES:
                    row += f" {data.get(metric, 0.0):>12.4f}"
                lines.append(row)

        lines.append("")
        lines.append("=" * 120)
        lines.append("Statistical significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        lines.append("Cohen's d interpretation: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")
        lines.append("All lower-is-better metrics. Negative diff = M4 better than baseline.")
        lines.append("=" * 120)

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_json_report(self, report: CrossSequenceReport, path: Path) -> None:
        """Write machine-readable JSON report."""
        data = {
            "analysis_mode": report.analysis_mode,
            "sequences": report.sequences,
            "methods": report.methods,
            "summaries": {},
            "pairwise_tests": [],
            "per_sequence": report.per_sequence,
        }

        for mid, s in report.summaries.items():
            data["summaries"][mid] = {
                "method_id": s.method_id,
                "n_sequences": s.n_sequences,
                "metrics": s.metrics,
                "wins": s.wins,
                "losses": s.losses,
                "ties": s.ties,
            }

        for t in report.pairwise_tests:
            data["pairwise_tests"].append(asdict(t))

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _write_per_sequence_csv(self, report: CrossSequenceReport, path: Path) -> None:
        """Write per-sequence metrics as CSV."""
        rows = []
        for seq in report.sequences:
            for mid in report.methods:
                data = report.per_sequence.get(seq, {}).get(mid, {})
                row = {"sequence": seq, "method": mid}
                for metric in METRIC_NAMES:
                    row[metric] = data.get(metric, "")
                rows.append(row)

        fieldnames = ["sequence", "method"] + METRIC_NAMES
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_pairwise_csv(self, report: CrossSequenceReport, path: Path) -> None:
        """Write pairwise test results as CSV."""
        if not report.pairwise_tests:
            return
        rows = [asdict(t) for t in report.pairwise_tests]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """CLI entry point for cross-sequence analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-sequence statistical analysis for IPF multi-video validation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="~/ipf_outputs",
        help="Root output directory containing run results",
    )
    parser.add_argument(
        "--run-prefix", type=str, default="multi_seq_",
        help="Prefix of run directories to include",
    )
    parser.add_argument(
        "--mode", type=str, default="steady", choices=["steady", "full"],
        help="Analysis mode: steady (skip warmup) or full",
    )
    args = parser.parse_args()

    analyzer = CrossSequenceAnalyzer(
        output_dir=args.output_dir,
        run_prefix=args.run_prefix,
        analysis_mode=args.mode,
    )

    print(f"Searching for runs in: {args.output_dir}")
    print(f"Run prefix: {args.run_prefix}")

    report = analyzer.analyze()

    print(f"\nAnalyzed {len(report.sequences)} sequences, {len(report.methods)} methods")
    print(f"Reports written to: {analyzer.output_dir / 'cross_sequence_analysis'}/")

    report_path = analyzer.output_dir / "cross_sequence_analysis" / "cross_sequence_report.txt"
    with open(report_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
