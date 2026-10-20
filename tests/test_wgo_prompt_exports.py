from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def string_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing {name} in {path}")


def text_fences(path: Path) -> list[str]:
    return [item.strip() for item in re.findall(r"```text\n(.*?)```", path.read_text(encoding="utf-8"), re.S)]


class WgoPromptExportTests(unittest.TestCase):
    def test_coarse_downloadable_prompt_matches_runtime(self) -> None:
        source = ROOT / "scripts/wgo_bench/contact_sheet_segment.py"
        exported = text_fences(ROOT / "prompts/wgo_contact_sheet_segmentation.md")
        self.assertEqual(exported[0], string_constant(source, "SYSTEM").strip())
        self.assertEqual(exported[1], string_constant(source, "USER_PROMPT_GEPA").strip())

    def test_s2_downloadable_prompt_matches_runtime(self) -> None:
        source = ROOT / "scripts/wgo_bench/seg_opt_ideas.py"
        runtime = string_constant(source, "USER_PROMPT_REFINE_FULL_COVER").strip()
        exported = text_fences(ROOT / "prompts/wgo_s2_fullcover_refine.md")
        self.assertEqual(
            exported[0],
            string_constant(ROOT / "scripts/wgo_bench/contact_sheet_segment.py", "SYSTEM").strip(),
        )
        self.assertEqual(exported[1], runtime)

    def test_selector_downloadable_prompt_matches_runtime(self) -> None:
        source = ROOT / "scripts/wgo_bench/select_pred_segment_candidates.py"
        runtime = string_constant(source, "PROMPT").strip()
        exported = text_fences(ROOT / "prompts/wgo_candidate_selector.md")
        self.assertEqual(exported[0], string_constant(source, "SYSTEM").strip())
        self.assertEqual(exported[1], runtime)

    def test_gemini_judge_downloadable_prompt_matches_runtime(self) -> None:
        runtime = string_constant(ROOT / "scripts/wgo_gemini_rescore.py", "JUDGE_RUBRIC").strip()
        exported = text_fences(ROOT / "prompts/wgo_gemini_judge.md")
        self.assertEqual(exported[0], runtime.replace("{{", "{").replace("}}", "}"))


if __name__ == "__main__":
    unittest.main()
