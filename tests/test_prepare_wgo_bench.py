import tempfile
import unittest
from pathlib import Path

from scripts.prepare_wgo_bench import parse_segments, subset_name, write_video


class PrepareWgoBenchTests(unittest.TestCase):
    def test_subset_name_uses_numeric_id_suffix(self):
        self.assertEqual(subset_name("homer_1"), "homer")
        self.assertEqual(subset_name("bridge_data_v2_12"), "bridge_data_v2")
        self.assertEqual(subset_name("unexpected"), "unknown")

    def test_parse_segments_accepts_json_text(self):
        value = '[{"start_sec": 1, "end_sec": 2.5, "subtask": "pick up cup"}]'
        self.assertEqual(
            parse_segments(value),
            [{"start_sec": 1.0, "end_sec": 2.5, "subtask": "pick up cup"}],
        )

    def test_write_video_accepts_hugging_face_bytes_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "videos" / "homer" / "homer_1.mp4"
            write_video(path, {"bytes": b"video-bytes", "path": None})
            self.assertEqual(path.read_bytes(), b"video-bytes")


if __name__ == "__main__":
    unittest.main()
