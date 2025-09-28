"""Tooling to generate pdf reports through latex"""

import abc
from collections.abc import Sequence
from pathlib import Path
import subprocess


class ReportElement(abc.ABC):  # pylint: disable=too-few-public-methods
    """Parent class for elements placed in a report object that generate latex code"""

    @abc.abstractmethod
    def latex(self) -> str:
        """Generate latex code"""


class ReportFigure(ReportElement):  # pylint: disable=too-few-public-methods
    """Figure element for reports"""

    def __init__(
        self,
        image_path: str,
        caption: str | None = None,
        width: float = 1.0,
    ):
        self.image_path = image_path
        self.caption = caption
        self.width = width

    def latex(self) -> str:
        caption = f"\\caption{{{self.caption}}}" if self.caption is not None else ""
        return (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f'    \\includegraphics[width={self.width:f}\\textwidth]{{"{self.image_path}"}}\n'
            f"    {caption}\n"
            f"\\end{{figure}}\n"
        )


class ReportTable(ReportElement):  # pylint: disable=too-few-public-methods
    """Table element for reports

    :param table_content: Two dimensional sequence of strings representing the table
        Content will be placed in a \\verb statement with the defined character being removed
    :param header: Header values of the table. If none are provided, no header is generated
    :param caption: Caption for the table
    :param cell_format: A sequence of latex format values. The default for a 4 column table is {cccc} resulting in centered tables
    :param horizontal_separator: If no cell format value is provided, this can be used to enable vertical lines.
    :param verb_char: Character used for the \\verb statements on table content. This character will be removed from the cell content strings.
    """

    def __init__(
        self,
        table_content: Sequence[Sequence[str]],
        header: Sequence[str] | None = None,
        caption: str | None = None,
        cell_format: str | None = None,
        horizontal_separator: str = " ",
        verb_char: str = "|",
    ):
        for row_idx, row in enumerate(table_content):
            if len(row) != len(table_content[0]):
                raise ValueError(
                    f"Row {row_idx} of table_content has different length."
                )
        self.table_content = table_content
        self.caption = caption
        self.format = (
            cell_format
            if cell_format is not None
            else horizontal_separator.join("c" for _ in range(len(table_content[0])))
        )
        self.verb_char = verb_char
        if header is None:
            self.header = ""
        else:
            if len(header) != len(table_content[0]):
                raise ValueError(
                    "Header length does not math table_content row length."
                )
            self.header = "\\hline\n" + " & ".join(header) + "\\\\"

    def latex(self) -> str:
        caption = f"\\caption{{{self.caption}}}" if self.caption is not None else ""
        table_content_str = ""
        for row in self.table_content:
            row = [
                "\\verb"
                + self.verb_char
                + cell_value.replace(self.verb_char, "")
                + self.verb_char
                for cell_value in row
            ]
            table_content_str += (" " * 4) + " & ".join(row) + "\\\\\n"

        return (
            f"\\begin{{table}}[h]\n"
            f"    \\centering\n"
            f"    {caption}\n"
            f"    \\begin{{tabular}}{{{self.format}}}\n"
            f"    {self.header}\n"
            f"    \\hline\n"
            f"{table_content_str}"
            f"    \\hline\n"
            f"    \\end{{tabular}}\n"
            f"\\end{{table}}\n"
        )


class Report(dict):
    """Latex code generator"""

    block_start = r"""\documentclass[12pt, a4paper]{report}
\usepackage[top=3cm, bottom=3cm, left = 2cm, right = 2cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{float}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{amsmath} % red arrow for listing line break

\lstset{
breaklines=true,
postbreak=\mbox{{$\hookrightarrow$}\space},
}


\begin{document}
"""

    block_end = r"\end{document}"

    sectioning_commands = [
        "chapter",
        "section",
        "subsection",
        "subsubsection",
        "paragraph",
        "subparagraph",
    ]

    @staticmethod
    def _generate_entry(entry):
        """Generate latex code for the given entry"""
        if isinstance(entry, str):
            return entry + "\n"
        if isinstance(entry, ReportElement):
            return entry.latex() + "\n"
        raise ValueError(
            f"Entry has unexpected type {type(entry)}. Only str, and ReportEntry are allowed."
        )

    def generate(self, structure: dict | None = None, level: int = 0) -> str:
        """Generate latex code"""
        if level >= len(self.sectioning_commands):
            raise ValueError("Maximum sectioning depth reached.")

        if structure is None:
            structure = self

        latex = ""
        for name, entry in structure.items():
            latex += f"\\{self.sectioning_commands[level]}{{{name}}}\n"

            if isinstance(entry, dict):
                latex += self.generate(entry, level + 1)
            elif isinstance(entry, list):
                for subentry in entry:
                    latex += self._generate_entry(subentry)
            else:
                latex += self._generate_entry(entry)

        if level == 0:
            latex = self.block_start + latex + self.block_end
        return latex

    def save(self, fname: str | Path) -> None:
        """Save latex code to file"""
        fname = Path(fname)
        with open(fname.with_suffix(".tex"), "w", encoding="UTF-8") as f:
            f.write(self.generate())

    def compile(self, fname: str | Path) -> Path:
        """Compile report using pdflatex
        If not present, the required .pdf and .tex suffixes are added as needed.

        returns the path of the generated pdf file
        """
        fname = Path(fname)
        self.save(fname.with_suffix(".tex"))
        retval = subprocess.run(
            ["pdflatex", "-halt-on-error", fname.with_suffix(".tex").name],
            cwd=fname.parent,
            check=False,
            capture_output=True,
        )
        if retval.returncode != 0:
            print(retval.stdout, "\n\n", retval.stderr)
            raise RuntimeError("PDF generation with pdflatex failed.")

        return fname.with_suffix(".pdf")
