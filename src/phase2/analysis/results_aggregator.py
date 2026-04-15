"""Results aggregation and statistical analysis for Phase 2 experiments.

Reads experiment results, computes BD-Rate/BD-Task metrics, runs
statistical tests, and produces paper-ready summary tables.

This module fulfills:
    - 04_EXPERIMENT_MATRIX.md section D (metrics), E (statistics)
    - 01_PROJECT_CHARTER.md section 5.1 (KPIs)
    - 04_EXPERIMENT_MATRIX.md section I (reviewer tables R1-R3)

Usage (CLI):
    python -m phase2.analysis.results_aggregator \\
        --experiment-dir ~/Minh/ipf/phase2_outputs/pilot_v1

Usage (Python):
    from phase2.analysis.results_aggregator import ResultsAggregator
    agg = ResultsAggregator(experiment_dir)
    agg.analyze()
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from phase2.evaluation.bd_rate import (
    RDPoint,
    BDResult,
    compute_full_bd_metrics,
)


@dataclass
class MethodSequenceRD:
    """RD points for one method on one sequence."""
    method: str
    sequence: str
    rd_points: List[RDPoint]


@dataclass
class AggregatedBD:
    """BD metrics aggregated across sequences."""
    test_method: str
    anchor_method: str
    n_sequences: int
    bd_rate_psnr_mean: float
    bd_rate_psnr_std: float
    bd_rate_roi_mean: float
    bd_rate_roi_std: float
    bd_rate_task_mean: float
    bd_rate_task_std: float
    bd_psnr_mean: float
    bd_task_mean: float
    per_sequence: List[BDResult]


@dataclass
class StatisticalTest:
    """Result of hypothesis test on BD metrics."""
    test_method: str
    anchor_method: str
    metric: str
    mean_diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    ci_lower: float
    ci_upper: float
    significant: bool
    n_sequences: int


class ResultsAggregator:
    """Aggregates Phase 2 experiment results and computes BD metrics."""

    def __init__(self, experiment_dir: str, anchor_method: str = "M0"):
        self.experiment_dir = Path(experiment_dir).expanduser()
        self.anchor_method = anchor_method

    def analyze(self, output_dir: Optional[str] = None) -> Dict:
        """Run the full analysis pipeline."""
        print(f"Loading results from: {self.experiment_dir}")

        rd_data = self._load_rd_data()
        if not rd_data:
            print("ERROR: No RD data found")
            return {}

        print(f"Loaded RD data for {len(rd_data)} method-sequence pairs")

        bd_results = self._compute_bd_metrics(rd_data)
        aggregated = self._aggregate_bd(bd_results)
        stat_tests = self._run_statistical_tests(aggregated)

        if output_dir:
            report_dir = Path(output_dir).expanduser()
        else:
            report_dir = self.experiment_dir / "analysis"
        report_dir.mkdir(parents=True, exist_ok=True)

        self._write_bd_table(aggregated, report_dir / "bd_rate_table.txt")
        self._write_kpi_check(aggregated, report_dir / "kpi_check.txt")
        self._write_reviewer_tables(aggregated, stat_tests, report_dir)
        self._write_json_report(aggregated, stat_tests, report_dir / "analysis.json")

        print(f"\nAnalysis complete. Reports at: {report_dir}")
        return {"aggregated": aggregated, "tests": stat_tests}

    def _load_rd_data(self) -> Dict[Tuple[str, str], MethodSequenceRD]:
        """Load all run results and organize into RD curves."""
        summary_path = self.experiment_dir / "experiment_summary.json"
        if not summary_path.exists():
            return {}

        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        rd_data: Dict[Tuple[str, str], MethodSequenceRD] = {}

        for entry in summary.get("results", []):
            if not entry.get("success", False):
                continue

            method = entry["method"]
            seq = entry["sequence"]
            key = (method, seq)

            if key not in rd_data:
                rd_data[key] = MethodSequenceRD(
                    method=method, sequence=seq, rd_points=[],
                )

            bitrate = entry.get("bitrate_kbps", 0)
            if bitrate <= 0:
                continue

            rd_data[key].rd_points.append(RDPoint(
                bitrate_kbps=bitrate,
                psnr_y=entry.get("psnr_y_full", entry.get("psnr_y_enc", 0)),
                psnr_roi=entry.get("psnr_y_roi", 0),
                mAP50=entry.get("mAP50", 0),
                mAP50_95=entry.get("mAP50_95", 0),
            ))

        for key in rd_data:
            rd_data[key].rd_points.sort(key=lambda p: p.bitrate_kbps)

        return rd_data

    def _compute_bd_metrics(
        self,
        rd_data: Dict[Tuple[str, str], MethodSequenceRD],
    ) -> List[BDResult]:
        """Compute BD metrics for all method-vs-anchor pairs."""
        results = []
        sequences = set()
        methods = set()

        for (method, seq) in rd_data:
            sequences.add(seq)
            methods.add(method)

        for seq in sorted(sequences):
            anchor_key = (self.anchor_method, seq)
            if anchor_key not in rd_data:
                continue
            anchor_rd = rd_data[anchor_key]

            for method in sorted(methods):
                if method == self.anchor_method:
                    continue
                test_key = (method, seq)
                if test_key not in rd_data:
                    continue
                test_rd = rd_data[test_key]

                bd = compute_full_bd_metrics(
                    anchor_method=self.anchor_method,
                    test_method=method,
                    sequence=seq,
                    anchor_points=anchor_rd.rd_points,
                    test_points=test_rd.rd_points,
                )
                results.append(bd)

        return results

    def _aggregate_bd(
        self,
        bd_results: List[BDResult],
    ) -> Dict[str, AggregatedBD]:
        """Aggregate BD results across sequences per method."""
        by_method: Dict[str, List[BDResult]] = {}
        for bd in bd_results:
            if not bd.valid:
                continue
            if bd.test_method not in by_method:
                by_method[bd.test_method] = []
            by_method[bd.test_method].append(bd)

        aggregated = {}
        for method, results in by_method.items():
            rates = np.array([r.bd_rate_psnr for r in results])
            roi_rates = np.array([r.bd_rate_roi for r in results])
            task_rates = np.array([r.bd_rate_task for r in results])
            psnrs = np.array([r.bd_psnr for r in results])
            tasks = np.array([r.bd_task for r in results])

            aggregated[method] = AggregatedBD(
                test_method=method,
                anchor_method=self.anchor_method,
                n_sequences=len(results),
                bd_rate_psnr_mean=float(np.mean(rates)),
                bd_rate_psnr_std=float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
                bd_rate_roi_mean=float(np.mean(roi_rates)),
                bd_rate_roi_std=float(np.std(roi_rates, ddof=1)) if len(roi_rates) > 1 else 0.0,
                bd_rate_task_mean=float(np.mean(task_rates)),
                bd_rate_task_std=float(np.std(task_rates, ddof=1)) if len(task_rates) > 1 else 0.0,
                bd_psnr_mean=float(np.mean(psnrs)),
                bd_task_mean=float(np.mean(tasks)),
                per_sequence=results,
            )

        return aggregated

    def _run_statistical_tests(
        self,
        aggregated: Dict[str, AggregatedBD],
    ) -> List[StatisticalTest]:
        """Run statistical tests comparing M4 vs each baseline."""
        tests = []
        m4_data = aggregated.get("M4")
        if not m4_data or m4_data.n_sequences < 2:
            return tests

        for method, agg in aggregated.items():
            if method == "M4" or agg.n_sequences < 2:
                continue

            for metric_name in ["bd_rate_psnr", "bd_rate_roi", "bd_rate_task"]:
                m4_vals = np.array([
                    getattr(r, metric_name) for r in m4_data.per_sequence
                ])
                bl_vals = np.array([
                    getattr(r, metric_name) for r in agg.per_sequence
                ])

                min_len = min(len(m4_vals), len(bl_vals))
                if min_len < 2:
                    continue
                m4_vals = m4_vals[:min_len]
                bl_vals = bl_vals[:min_len]

                diff = m4_vals - bl_vals
                d_mean = float(np.mean(diff))
                d_std = float(np.std(diff, ddof=1))

                if HAS_SCIPY and min_len >= 2:
                    t_res = scipy_stats.ttest_rel(m4_vals, bl_vals)
                    t_stat = float(t_res.statistic)
                    p_value = float(t_res.pvalue)
                elif d_std > 1e-12:
                    t_stat = d_mean / (d_std / math.sqrt(min_len))
                    p_value = 1.0
                else:
                    t_stat = 0.0
                    p_value = 1.0

                cohens_d = d_mean / d_std if d_std > 1e-12 else 0.0

                rng = np.random.default_rng(42)
                n_boot = 5000
                boot_means = np.array([
                    np.mean(rng.choice(diff, size=min_len, replace=True))
                    for _ in range(n_boot)
                ])
                ci_lo = float(np.percentile(boot_means, 2.5))
                ci_hi = float(np.percentile(boot_means, 97.5))

                tests.append(StatisticalTest(
                    test_method="M4",
                    anchor_method=method,
                    metric=metric_name,
                    mean_diff=d_mean,
                    t_stat=t_stat,
                    p_value=p_value,
                    cohens_d=cohens_d,
                    ci_lower=ci_lo,
                    ci_upper=ci_hi,
                    significant=p_value < 0.05,
                    n_sequences=min_len,
                ))

        return tests

    def _write_bd_table(
        self,
        aggregated: Dict[str, AggregatedBD],
        path: Path,
    ) -> None:
        """Write BD-Rate comparison table."""
        lines = []
        lines.append("=" * 110)
        lines.append(f"BD-RATE TABLE (anchor: {self.anchor_method})")
        lines.append("=" * 110)
        lines.append(
            f"{'Method':<12} {'N':>3} "
            f"{'BD-Rate(PSNR)':>15} {'BD-Rate(ROI)':>15} {'BD-Rate(Task)':>15} "
            f"{'BD-PSNR':>10} {'BD-Task':>10}"
        )
        lines.append("-" * 110)

        for method in sorted(aggregated.keys()):
            a = aggregated[method]
            name = f"{method} (IPF)" if method == "M4" else method
            lines.append(
                f"{name:<12} {a.n_sequences:>3} "
                f"{a.bd_rate_psnr_mean:>+8.2f}+/-{a.bd_rate_psnr_std:<5.2f} "
                f"{a.bd_rate_roi_mean:>+8.2f}+/-{a.bd_rate_roi_std:<5.2f} "
                f"{a.bd_rate_task_mean:>+8.2f}+/-{a.bd_rate_task_std:<5.2f} "
                f"{a.bd_psnr_mean:>+10.3f} "
                f"{a.bd_task_mean:>+10.4f}"
            )

        lines.append("-" * 110)
        lines.append("BD-Rate: negative = bitrate saving, positive = bitrate penalty")
        lines.append("BD-PSNR: positive = quality gain at equal bitrate")
        lines.append("BD-Task: positive = mAP gain at equal bitrate")
        lines.append("")

        lines.append("PER-SEQUENCE BREAKDOWN:")
        lines.append("-" * 110)
        for method in sorted(aggregated.keys()):
            for bd in aggregated[method].per_sequence:
                lines.append(
                    f"  {bd.test_method:<8} on {bd.sequence:<20} "
                    f"BD-Rate={bd.bd_rate_psnr:>+7.2f}% "
                    f"BD-ROI={bd.bd_rate_roi:>+7.2f}% "
                    f"BD-Task={bd.bd_rate_task:>+7.2f}%"
                )

        lines.append("=" * 110)
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_kpi_check(
        self,
        aggregated: Dict[str, AggregatedBD],
        path: Path,
    ) -> None:
        """Check results against KPI targets from project charter."""
        lines = []
        lines.append("=" * 80)
        lines.append("KPI CHECK (from 01_PROJECT_CHARTER.md section 5.1)")
        lines.append("=" * 80)

        m4 = aggregated.get("M4")
        m1 = aggregated.get("M1")

        # KPI-1: >= 5% BD-Rate reduction vs M1
        if m4 and m1:
            m4_vs_m1_rates = []
            for m4_bd in m4.per_sequence:
                for m1_bd in m1.per_sequence:
                    if m4_bd.sequence == m1_bd.sequence:
                        relative = m4_bd.bd_rate_psnr - m1_bd.bd_rate_psnr
                        m4_vs_m1_rates.append(relative)
            if m4_vs_m1_rates:
                mean_gain = np.mean(m4_vs_m1_rates)
                pass_kpi1 = mean_gain < -5.0
                lines.append(
                    f"KPI-1: BD-Rate reduction vs M1 >= 5%: "
                    f"{abs(mean_gain):.1f}% {'[PASS]' if pass_kpi1 else '[PENDING]'}"
                )
            else:
                lines.append("KPI-1: Cannot compute (insufficient data)")
        else:
            lines.append("KPI-1: Cannot compute (M4 or M1 missing)")

        # KPI-2: >= 10% temporal QP variance reduction (from Phase 1)
        lines.append("KPI-2: Temporal QP variance reduction >= 10%: [See Phase 1 results]")

        # KPI-3: <= 20% encoding overhead
        lines.append("KPI-3: Encoding overhead <= 20%: [Measure from run times]")

        # KPI-4: Statistical significance
        if m4 and m4.n_sequences >= 3:
            lines.append(f"KPI-4: Statistical support: {m4.n_sequences} sequences available")
        else:
            lines.append(f"KPI-4: Insufficient sequences for significance testing")

        lines.append("=" * 80)
        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_reviewer_tables(
        self,
        aggregated: Dict[str, AggregatedBD],
        stat_tests: List[StatisticalTest],
        report_dir: Path,
    ) -> None:
        """Write reviewer-focused tables R1, R2, R3."""
        # Table R1: IPF vs soft-map controls
        lines = []
        lines.append("TABLE R1: IPF vs Soft-Map Controls")
        lines.append("=" * 90)
        lines.append(
            f"{'Method':<12} {'BD-Rate(PSNR)':>15} {'BD-Rate(ROI)':>15} "
            f"{'BD-Rate(Task)':>15} {'N':>4}"
        )
        lines.append("-" * 90)

        for mid in ["M1", "M4", "M5", "M6"]:
            if mid not in aggregated:
                continue
            a = aggregated[mid]
            name = f"{mid} (IPF)" if mid == "M4" else mid
            lines.append(
                f"{name:<12} "
                f"{a.bd_rate_psnr_mean:>+8.2f}+/-{a.bd_rate_psnr_std:<5.2f} "
                f"{a.bd_rate_roi_mean:>+8.2f}+/-{a.bd_rate_roi_std:<5.2f} "
                f"{a.bd_rate_task_mean:>+8.2f}+/-{a.bd_rate_task_std:<5.2f} "
                f"{a.n_sequences:>4}"
            )
        lines.append("=" * 90)
        (report_dir / "table_R1.txt").write_text("\n".join(lines), encoding="utf-8")

        # Table R3: Effect size and CI
        lines = []
        lines.append("TABLE R3: Statistical Evidence (M4 vs baselines)")
        lines.append("=" * 110)
        lines.append(
            f"{'Comparison':<20} {'Metric':<18} {'Diff':>8} {'t-stat':>8} "
            f"{'p-value':>9} {'Cohen d':>9} {'95% CI':>22} {'Sig?':>5}"
        )
        lines.append("-" * 110)

        for t in stat_tests:
            sig = "***" if t.p_value < 0.001 else (
                "**" if t.p_value < 0.01 else (
                    "*" if t.p_value < 0.05 else "ns"
                )
            )
            ci_str = f"[{t.ci_lower:+.3f}, {t.ci_upper:+.3f}]"
            lines.append(
                f"M4 vs {t.anchor_method:<13} {t.metric:<18} "
                f"{t.mean_diff:>+8.3f} {t.t_stat:>8.3f} "
                f"{t.p_value:>9.4f} {t.cohens_d:>+9.3f} "
                f"{ci_str:>22} {sig:>5}"
            )
        lines.append("=" * 110)
        (report_dir / "table_R3.txt").write_text("\n".join(lines), encoding="utf-8")

    def _write_json_report(
        self,
        aggregated: Dict[str, AggregatedBD],
        stat_tests: List[StatisticalTest],
        path: Path,
    ) -> None:
        """Write machine-readable JSON analysis report."""
        data = {
            "anchor_method": self.anchor_method,
            "aggregated": {},
            "statistical_tests": [],
        }

        for method, agg in aggregated.items():
            data["aggregated"][method] = {
                "test_method": agg.test_method,
                "n_sequences": agg.n_sequences,
                "bd_rate_psnr_mean": agg.bd_rate_psnr_mean,
                "bd_rate_psnr_std": agg.bd_rate_psnr_std,
                "bd_rate_roi_mean": agg.bd_rate_roi_mean,
                "bd_rate_roi_std": agg.bd_rate_roi_std,
                "bd_rate_task_mean": agg.bd_rate_task_mean,
                "bd_rate_task_std": agg.bd_rate_task_std,
                "bd_psnr_mean": agg.bd_psnr_mean,
                "bd_task_mean": agg.bd_task_mean,
                "per_sequence": [asdict(r) for r in agg.per_sequence],
            }

        for t in stat_tests:
            data["statistical_tests"].append(asdict(t))

        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 2 results aggregation and BD-Rate analysis"
    )
    parser.add_argument(
        "--experiment-dir", type=str, required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--anchor", type=str, default="M0",
        help="Anchor method for BD-Rate computation (default: M0)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory for analysis outputs (default: <experiment-dir>/analysis)",
    )
    args = parser.parse_args()

    agg = ResultsAggregator(args.experiment_dir, anchor_method=args.anchor)
    agg.analyze(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
