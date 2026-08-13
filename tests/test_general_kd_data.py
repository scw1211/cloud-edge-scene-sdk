import unittest

from edge_llm_factory.contracts import ManifestError
from edge_llm_factory.general_kd_data import resolve_required_categories


class GeneralKdDataCategoryTests(unittest.TestCase):
    def test_infers_only_categories_present_in_focused_source(self):
        rows = [
            {"category": "natural_language_reasoning"},
            {"category": "natural_language_reasoning"},
        ]
        self.assertEqual(
            resolve_required_categories([], rows),
            ("natural_language_reasoning",),
        )

    def test_explicit_focused_category_is_accepted(self):
        self.assertEqual(
            resolve_required_categories(
                ["natural_language_reasoning"],
                [{"category": "natural_language_reasoning"}],
            ),
            ("natural_language_reasoning",),
        )

    def test_rejects_duplicate_or_undeclared_categories(self):
        with self.assertRaisesRegex(ManifestError, "不能重复"):
            resolve_required_categories(
                ["math", "math"], [{"category": "math"}]
            )
        with self.assertRaisesRegex(ManifestError, "未声明"):
            resolve_required_categories(
                ["math"],
                [
                    {"category": "math"},
                    {"category": "natural_language_reasoning"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
