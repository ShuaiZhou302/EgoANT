import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "merge_judge_segments.py"
SPEC = importlib.util.spec_from_file_location("merge_judge_segments", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def segment(seg_id: int, frames: int = 30) -> dict:
    return {
        "seg_id": seg_id,
        "start_frame": seg_id * frames,
        "end_frame": (seg_id + 1) * frames,
        "instruction": f"Action {seg_id}",
        "actor": "right hand",
        "verb": "move",
        "main_object": {
            "name": "object",
            "name_candidates": [{"name": "object", "confidence": 0.8}],
            "confidence": 0.8,
        },
        "involved_objects": [],
        "contact": "visible contact",
        "final_state": "object moved",
        "confidence": 0.8,
        "needs_review": False,
    }


def boundary(left: int, right: int, confidence: float = 0.9) -> dict:
    vote = {
        "merge": True,
        "confidence": confidence,
        "reason": "one continuous action",
    }
    return {
        "left_seg_id": left,
        "right_seg_id": right,
        "merge": True,
        "votes": [vote],
        "confidence": confidence,
    }


class MergeJudgeTests(unittest.TestCase):
    def test_candidate_groups_follow_connected_boundaries(self) -> None:
        segments = [segment(i) for i in range(5)]
        boundaries = {
            (0, 1): boundary(0, 1),
            (1, 2): boundary(1, 2),
            (2, 3): {"merge": False},
            (3, 4): boundary(3, 4),
        }
        groups = MODULE.build_candidate_groups(segments, boundaries)
        self.assertEqual([[s["seg_id"] for s in group] for group in groups], [[0, 1, 2], [3, 4]])

    def test_oversized_group_splits_at_weakest_boundaries(self) -> None:
        segments = [segment(i) for i in range(16)]
        boundaries = {(i, i + 1): boundary(i, i + 1, confidence=0.9) for i in range(15)}
        boundaries[(7, 8)] = boundary(7, 8, confidence=0.76)
        groups = MODULE.split_oversized_group(
            segments,
            boundaries,
            fps=30.0,
            max_segments=8,
            max_duration_sec=15.0,
        )
        self.assertEqual([[s["seg_id"] for s in group] for group in groups], [list(range(8)), list(range(8, 16))])

    def test_duration_limit_also_splits_candidate_group(self) -> None:
        segments = [segment(i, frames=120) for i in range(4)]
        boundaries = {(i, i + 1): boundary(i, i + 1, confidence=0.9) for i in range(3)}
        boundaries[(1, 2)] = boundary(1, 2, confidence=0.76)
        groups = MODULE.split_oversized_group(
            segments,
            boundaries,
            fps=30.0,
            max_segments=8,
            max_duration_sec=10.0,
        )
        self.assertEqual([[s["seg_id"] for s in group] for group in groups], [[0, 1], [2, 3]])

    def test_second_stage_uses_new_merged_annotation(self) -> None:
        segments = [segment(0), segment(1)]
        boundaries = {(0, 1): boundary(0, 1)}
        rewritten = {
            "merge": True,
            "reason": "one continuous placement action",
            "annotation": {
                "instruction": "Place the potato into the bag.",
                "alternatives": ["a", "b", "c", "d", "e"],
                "actor": "right hand",
                "verb": "place",
                "main_object": {
                    "name": "potato",
                    "name_candidates": [{"name": "potato", "confidence": 0.9}],
                    "confidence": 0.9,
                },
                "involved_objects": [{"name": "bag", "role": "container"}],
                "object_aliases": ["brown object", "potato"],
                "contact": "grasp",
                "final_state": "potato is in bag",
                "confidence": 0.9,
                "needs_review": False,
            },
        }
        args = SimpleNamespace(
            base_url="test",
            rewrite_model="test",
            api_key_env="TEST_KEY",
            timeout=1.0,
            max_tokens=65536,
            api_retries=0,
            api_image_max_side=640,
            api_image_jpeg_quality=85,
            disable_thinking=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(MODULE, "call_rewrite_api", return_value=rewritten):
            result = MODULE.finalize_candidate_group(
                segments, boundaries, Path(temp_dir), "overlay", Path(temp_dir) / "aux", args
            )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["instruction"], "Place the potato into the bag.")
        self.assertEqual(result[0]["object_aliases"], ["brown object", "potato"])
        self.assertEqual(result[0]["source_seg_ids"], [0, 1])

    def test_second_stage_accepts_three_alternatives(self) -> None:
        result = MODULE.validate_rewrite_result(
            {
                "merge": True,
                "reason": "one action",
                "annotation": {
                    "instruction": "Place the potato into the bag.",
                    "alternatives": ["Place the potato in the bag.", "Put the potato into the bag.", "Insert the potato into the bag."],
                    "actor": "right hand",
                    "verb": "place",
                    "main_object": {
                        "name": "potato",
                        "name_candidates": [{"name": "potato", "confidence": 0.9}],
                        "confidence": 0.9,
                    },
                    "involved_objects": [{"name": "bag", "role": "container"}],
                    "object_aliases": ["brown object"],
                    "contact": "grasp",
                    "final_state": "potato is in bag",
                    "confidence": 0.9,
                    "needs_review": False,
                },
            }
        )
        self.assertTrue(result["merge"])

    def test_rejected_second_stage_recursively_restores_boundaries(self) -> None:
        segments = [segment(0), segment(1), segment(2)]
        boundaries = {
            (0, 1): boundary(0, 1, confidence=0.9),
            (1, 2): boundary(1, 2, confidence=0.76),
        }
        args = SimpleNamespace(
            base_url="test",
            rewrite_model="test",
            api_key_env="TEST_KEY",
            timeout=1.0,
            max_tokens=65536,
            api_retries=0,
            api_image_max_side=640,
            api_image_jpeg_quality=85,
            disable_thinking=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            MODULE, "call_rewrite_api", return_value={"merge": False, "reason": "not one atomic action"}
        ):
            result = MODULE.finalize_candidate_group(
                segments, boundaries, Path(temp_dir), "overlay", Path(temp_dir) / "aux", args
            )
        self.assertEqual([item["source_seg_ids"] for item in result], [[0], [1], [2]])

    def test_formal_annotations_hide_debug_fields_and_reindex(self) -> None:
        segments = [
            {
                "seg_id": 12,
                "source_seg_ids": [4, 5],
                "start_frame": 100,
                "end_frame": 180,
                "instruction": "Place the potato into the bag.",
                "main_object": {"name": "potato", "name_candidates": [], "confidence": 0.9},
                "merge_reason": "accepted by rewrite judge",
            }
        ]
        formal = MODULE.formal_annotations(
            {"video_path": "example.mp4", "fps": 30, "nframes": 180},
            segments,
            fps=30.0,
        )
        self.assertEqual(list(formal)[:3], ["fps", "nframes", "duration_sec"])
        self.assertNotIn("video_path", formal)
        self.assertEqual(formal["duration_sec"], 6.0)
        self.assertEqual(list(formal["segments"][0])[:3], ["seg_id", "start_frame", "end_frame"])
        self.assertEqual(formal["segments"][0]["seg_id"], 0)
        self.assertNotIn("source_seg_ids", formal["segments"][0])
        self.assertNotIn("merge_reason", formal["segments"][0])

    def test_pure_static_hint_requires_no_dynamic_action(self) -> None:
        static = segment(0)
        static.update({"verb": "hold", "instruction": "Hold the bowl steady."})
        moving = segment(1)
        moving.update({"verb": "hold", "instruction": "Hold and move the bowl forward."})
        self.assertTrue(MODULE.is_pure_static_segment(static))
        self.assertFalse(MODULE.is_pure_static_segment(moving))

    def test_rewrite_normalizes_second_person_and_transfer(self) -> None:
        result = MODULE.validate_rewrite_result(
            {
                "merge": True,
                "reason": "one action",
                "annotation": {
                    "instruction": "Transfer the spoon with your right hand.",
                    "alternatives": ["Transfer it.", "Move it.", "Place it."],
                    "actor": "right hand",
                    "verb": "transfer",
                    "main_object": {"name": "spoon", "name_candidates": [], "confidence": 0.9},
                    "involved_objects": [],
                    "object_aliases": [],
                    "contact": "grasp",
                    "final_state": "spoon moved",
                    "confidence": 0.9,
                    "needs_review": False,
                },
            }
        )
        self.assertEqual(result["annotation"]["instruction"], "Move the spoon with the right hand.")
        self.assertEqual(result["annotation"]["verb"], "move")


if __name__ == "__main__":
    unittest.main()
