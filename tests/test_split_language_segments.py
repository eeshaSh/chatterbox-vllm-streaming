import unittest

from chatterbox_vllm.text_utils import split_language_segments


class TestSplitLanguageSegments(unittest.TestCase):
    def test_no_tags_single_segment(self):
        self.assertEqual(
            split_language_segments("Hej och välkommen.", "sv"),
            [("sv", "Hej och välkommen.")],
        )

    def test_single_english_span(self):
        self.assertEqual(
            split_language_segments("Ladda vid en <en>Supercharger</en> nu.", "sv"),
            [("sv", "Ladda vid en "), ("en", "Supercharger"), ("sv", " nu.")],
        )

    def test_tag_at_start(self):
        self.assertEqual(
            split_language_segments("<en>Supercharger</en> finns i Malmö.", "sv"),
            [("en", "Supercharger"), ("sv", " finns i Malmö.")],
        )

    def test_tag_at_end(self):
        self.assertEqual(
            split_language_segments("Prova vår <en>Webshop</en>", "sv"),
            [("sv", "Prova vår "), ("en", "Webshop")],
        )

    def test_multiple_spans(self):
        self.assertEqual(
            split_language_segments("Se <en>Supercharger</en> och <en>Webshop</en>.", "sv"),
            [
                ("sv", "Se "),
                ("en", "Supercharger"),
                ("sv", " och "),
                ("en", "Webshop"),
                ("sv", "."),
            ],
        )

    def test_case_insensitive_tag(self):
        self.assertEqual(
            split_language_segments("x <EN>Supercharger</EN> y", "sv"),
            [("sv", "x "), ("en", "Supercharger"), ("sv", " y")],
        )

    def test_unknown_code_left_as_literal(self):
        # <zz> is not a supported language — treat as literal text, strip stray tag
        self.assertEqual(
            split_language_segments("hej <zz>foo</zz> då", "sv"),
            [("sv", "hej foo då")],
        )

    def test_stray_unmatched_tag_stripped(self):
        self.assertEqual(
            split_language_segments("hej <en> då", "sv"),
            [("sv", "hej  då")],
        )

    def test_base_language_respected(self):
        self.assertEqual(
            split_language_segments("Åbn <en>Software Upgrades</en> menuen.", "da"),
            [("da", "Åbn "), ("en", "Software Upgrades"), ("da", " menuen.")],
        )

    def test_empty_spans_dropped(self):
        self.assertEqual(
            split_language_segments("a <en></en> b", "sv"),
            [("sv", "a "), ("sv", " b")],
        )


if __name__ == "__main__":
    unittest.main()
