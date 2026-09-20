import unittest

from scripts.add_episode_annotation import build_prompt, compact_instructions, with_episode


class EpisodeAnnotationTests(unittest.TestCase):
    def test_compacts_only_consecutive_duplicates(self):
        segments = [
            {"instruction": "Place the cup.", "start_frame": 0, "end_frame": 10},
            {"instruction": "Place the cup.", "start_frame": 10, "end_frame": 20},
            {"instruction": "Lift the cup.", "start_frame": 20, "end_frame": 30},
            {"instruction": "N/A", "start_frame": 30, "end_frame": 40},
        ]
        compact = compact_instructions(segments)
        self.assertEqual(len(compact), 2)
        self.assertEqual(compact[0]["repeat_count"], 2)
        self.assertEqual(compact[0]["end_frame"], 20)

    def test_episode_field_precedes_segments(self):
        output = with_episode({"fps": 30, "segments": []}, {"instruction": "Task"})
        self.assertEqual(list(output), ["fps", "episode_annotation", "segments"])
        self.assertEqual(output["episode_annotation"], {"instruction": "Task"})

    def test_prompt_requests_short_task_instruction(self):
        prompt = build_prompt([{"instruction": "Place an item into a bag."}])
        self.assertIn("usually 4 to 12 words", prompt)
        self.assertIn("Do not start with 'The person'", prompt)
        self.assertIn("not the sequence of steps", prompt)


if __name__ == "__main__":
    unittest.main()
