from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "wgo_bench" / "assemble_candidate_labels.py"
SPEC = importlib.util.spec_from_file_location("assemble_candidate_labels", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class AssembleCandidateLabelsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boundary = {
            "id": "homer_1",
            "segments": [{"start_sec": 0.0, "end_sec": 2.0, "subtask": "seed"}],
        }

    def test_assembles_named_labels(self) -> None:
        raw = {"segments": [{"start_sec": 0.0, "end_sec": 2.0, "subtask": "open drawer"}]}
        result = MODULE.assemble_episode(self.boundary, [("raw", raw)])
        self.assertEqual(result["segments"][0]["candidate_labels"], {"raw": "open drawer"})

    def test_rejects_boundary_mismatch(self) -> None:
        raw = {"segments": [{"start_sec": 0.2, "end_sec": 2.0, "subtask": "open drawer"}]}
        with self.assertRaisesRegex(ValueError, "boundary mismatch"):
            MODULE.assemble_episode(self.boundary, [("raw", raw)])


if __name__ == "__main__":
    unittest.main()
