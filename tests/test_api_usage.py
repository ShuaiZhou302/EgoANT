import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "api_usage.py"
SPEC = importlib.util.spec_from_file_location("api_usage", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ApiUsageTests(unittest.TestCase):
    def test_server_usage_is_aggregated_by_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            events = Path(temp_dir) / "api_usage.jsonl"
            output = Path(temp_dir) / "token_usage.json"
            for stage in ("caption", "merge_judge"):
                MODULE.append_usage(
                    events,
                    MODULE.usage_event(
                        {"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
                        stage=stage,
                        model="test-model",
                        latency_sec=1.5,
                        success=True,
                        attempt=1,
                    ),
                )
            summary = MODULE.summarize_usage(events, output)
            self.assertEqual(summary["requests"], 2)
            self.assertEqual(summary["total_tokens"], 240)
            self.assertEqual(summary["stages"]["caption"]["prompt_tokens"], 100)
            self.assertEqual(json.loads(output.read_text())["successful_requests"], 2)

    def test_missing_server_usage_is_counted_without_fake_tokens(self) -> None:
        event = MODULE.usage_event(
            None,
            stage="caption",
            model="test-model",
            latency_sec=0.5,
            success=False,
            attempt=2,
            error="rate limited",
        )
        self.assertFalse(event["usage_reported"])
        self.assertIsNone(event["total_tokens"])


if __name__ == "__main__":
    unittest.main()
