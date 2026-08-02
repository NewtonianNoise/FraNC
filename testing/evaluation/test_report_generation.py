"""Tests for the report generators"""

import unittest

from franc.evaluation.report_generation import (
    LatexReport,
    HTMLReport,
    _escape_latex_chars,
)


class TestReportEscaping(unittest.TestCase):
    """Tests for the escaping of special characters"""

    def test_latex_special_characters(self):
        """check that latex special characters are escaped exactly once"""
        self.assertEqual(
            _escape_latex_chars(r"a\b_c~d"),
            r"a\textbackslash{}b\_c\textasciitilde{}d",
        )

    def test_section_names_are_escaped(self):
        """check that section names are escaped in both report formats"""
        self.assertIn(r"{a\_b}", LatexReport({"a_b": []}).generate())
        self.assertIn("&lt;a&gt;", HTMLReport({"<a>": []}).generate())
