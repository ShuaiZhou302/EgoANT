from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wgo_bench" / "wgo_metrics.py"
SPEC = importlib.util.spec_from_file_location("wgo_metrics", MODULE_PATH)
assert SPEC and SPEC.loader
METRICS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = METRICS
SPEC.loader.exec_module(METRICS)


class WgoMetricsTests(unittest.TestCase):
    def test_perfect_match(self) -> None:
        segments = [METRICS.Segment(0, 2), METRICS.Segment(2, 4)]
        result = METRICS.segment_f1(segments, segments)
        self.assertEqual(result["n_match"], 2)
        self.assertEqual(result["f1"], 1.0)

    def test_outer_snap_changes_only_episode_edges(self) -> None:
        gold = [METRICS.Segment(0, 2), METRICS.Segment(2, 4)]
        pred = [METRICS.Segment(0.5, 2), METRICS.Segment(2, 3.5)]
        snapped = METRICS.snap_outer_boundaries(pred, gold)
        self.assertEqual((snapped[0].start, snapped[-1].end), (0, 4))
        self.assertEqual((snapped[0].end, snapped[-1].start), (2, 2))

    def test_greedy_matching_is_one_to_one(self) -> None:
        gold = [METRICS.Segment(0, 4)]
        pred = [METRICS.Segment(0, 4), METRICS.Segment(0, 4)]
        result = METRICS.segment_f1(pred, gold, outer_snap=False)
        self.assertEqual(result["n_match"], 1)
        self.assertAlmostEqual(result["precision"], 0.5)

    def test_iou_threshold_is_inclusive(self) -> None:
        gold = [METRICS.Segment(0, 4)]
        pred = [METRICS.Segment(0, 3)]
        result = METRICS.segment_f1(pred, gold, threshold=0.75, outer_snap=False)
        self.assertEqual(result["n_match"], 1)


if __name__ == "__main__":
    unittest.main()
