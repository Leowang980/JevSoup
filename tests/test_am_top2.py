"""CPU-only checks for the isolated AM Top-2 adaptation."""
import unittest

from jev_lora.models import select_weights
from scripts.run_am_top2_4b import complete_two, parse_two


class AMTop2Test(unittest.TestCase):
    def test_parses_distinct_ids_in_rank_order(self):
        names = ["rte", "boolq", "arc_easy"]
        self.assertEqual(parse_two("`ARC_EASY`, rte\nextra", names),
                         ("arc_easy", "rte"))
        self.assertEqual(parse_two("rte, rte, boolq", names),
                         ("rte", "boolq"))
        self.assertEqual(parse_two("unknown", names), ())

    def test_keyword_fallback_fills_only_missing_choice(self):
        cards = [{"id": "rte", "description": "textual entailment"},
                 {"id": "boolq", "description": "answer passage question"},
                 {"id": "sciq", "description": "science physics biology"}]
        choices, scores = complete_two(("rte",), "science physics", cards, 42)
        self.assertEqual(choices, ("rte", "sciq"))
        self.assertIsNotNone(scores)
        choices, scores = complete_two(("rte", "sciq"), "anything", cards, 42)
        self.assertEqual(choices, ("rte", "sciq"))
        self.assertIsNone(scores)

    def test_generic_mixer_selects_equal_top_two(self):
        row = {"input_hash": "example"}
        route = {"kind": "am", "input_hash": "example",
                 "scores": {"rte": 2.0, "boolq": 1.0, "sciq": 0.0}}
        self.assertEqual(select_weights("am-top2-equal", row, route,
                                        list(route["scores"])),
                         {"rte": 0.5, "boolq": 0.5})


if __name__ == "__main__":
    unittest.main()
