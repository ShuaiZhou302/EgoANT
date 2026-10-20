from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/wgo_bench/make_episode_inputs.py"
SPEC = importlib.util.spec_from_file_location("make_episode_inputs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EpisodeInputTests(unittest.TestCase):
    def test_gold_boundaries_and_labels_are_removed(self) -> None:
        annotation = {
            "id": "homer_1",
            "instruction": "open the drawer",
            "segments": [
                {"start_sec": 1.0, "end_sec": 2.0, "subtask": "gold label"}
            ],
            "metadata": {"private": "unused"},
        }
        self.assertEqual(
            MODULE.public_episode_context(annotation),
            {"id": "homer_1", "instruction": "open the drawer"},
        )


if __name__ == "__main__":
    unittest.main()
