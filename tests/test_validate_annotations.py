import unittest

from scripts.validate_annotations import is_api_fallback


class ValidateAnnotationsTests(unittest.TestCase):
    def test_detects_api_fallback_signature(self):
        self.assertTrue(is_api_fallback({"instruction": "N/A", "confidence": 0.0, "needs_review": True}))

    def test_does_not_reject_other_na_annotations(self):
        self.assertFalse(is_api_fallback({"instruction": "N/A", "confidence": 0.2, "needs_review": True}))
        self.assertFalse(is_api_fallback({"instruction": "N/A", "confidence": 0.0, "needs_review": False}))


if __name__ == "__main__":
    unittest.main()
