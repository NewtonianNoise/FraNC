"""Tooling to generate pdf reports through latex"""

import abc
from collections.abc import Sequence
from pathlib import Path
import base64
import subprocess


class ReportElement(abc.ABC):  # pylint: disable=too-few-public-methods
    """Parent class for elements placed in a report object that generate latex code"""

    @abc.abstractmethod
    def latex(self) -> str:
        """Generate latex code"""

    @abc.abstractmethod
    def html(self) -> str:
        """Generate latex code"""


class ReportFigure(ReportElement):  # pylint: disable=too-few-public-methods
    """Figure element for reports

    :param image_path: Image path within the project location
    :param base_path: Project location
    :param caption: Optional figure caption
    :param width: Optional display width of the figure
    """

    def __init__(
        self,
        image_path: str,
        base_path: str | Path,
        caption: str | None = None,
        width: float = 1.0,
    ):
        self.image_path = image_path
        self.base_path = base_path
        self.caption = caption
        self.width = width

    def latex(self) -> str:
        caption = f"\\caption{{{self.caption}}}" if self.caption is not None else ""
        return (
            f"\\begin{{figure}}[H]\n"
            f"    \\centering\n"
            f'    \\includegraphics[width={self.width:f}\\textwidth]{{"{"../"+self.image_path}"}}\n'
            f"    {caption}\n"
            f"\\end{{figure}}\n"
        )

    def html(self) -> str:
        caption = (
            f"<figcaption>{self.caption}</figcaption>"
            if self.caption is not None
            else ""
        )
        with open(Path(self.base_path) / Path("report") / self.image_path, "rb") as f:
            file_data = f.read()

        file_type = self.image_path.split(".")[-1]
        if file_type == "pdf":
            file_type = "application/" + file_type
        else:
            file_type = "image/" + file_type
        img_b64 = f"data:{file_type};base64," + base64.b64encode(file_data).decode()

        return (
            f"<figure>\n"
            f'    <iframe width="100%" style="aspect-ratio: 2" src="{img_b64}"></iframe>\n'
            f"    {caption}\n"
            f"</figure>\n"
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

    def html(self) -> str:
        caption = (
            f"<figcaption>{self.caption}</figcaption>"
            if self.caption is not None
            else ""
        )
        table_content_str = ""
        for row in self.table_content:
            row = [f"    <th>{cell_value}</th>" for cell_value in row]
            table_content_str += "  <tr>\n" + "\n".join(row) + "\n  </tr>\n"

        return (
            f"<figure>\n"
            f"    {caption}\n"
            f"    <table>{table_content_str}</table>\n"
            f"</figure>\n"
        )


class ReportCodeListing(ReportElement):  # pylint: disable=too-few-public-methods
    """Table element for reports

    :param content: The code to be displayed
    """

    def __init__(
        self,
        content: str,
    ):
        self.content = content

    def latex(self) -> str:

        return (
            f"\\begin{{lstlisting}}\n" f"    {self.content}\n" f"\\end{{lstlisting}}\n"
        )

    def html(self) -> str:
        return f"<pre><code>\n" f"    {self.content}\n" f"</code></pre>\n"


class Report(dict, abc.ABC):
    """Base class for report generators"""

    @property
    @abc.abstractmethod
    def file_ending(self) -> str:
        """file ending for generated files"""

    @abc.abstractmethod
    def generate(self, structure: dict | None = None, level: int = 0) -> str:
        """Generate report code"""

    def save(self, fname: str | Path) -> None:
        """Save report code to file"""
        fname = Path(fname)
        with open(
            fname.with_suffix("." + self.file_ending), "w", encoding="UTF-8"
        ) as f:
            f.write(self.generate())

    def compile(self, fname: str | Path) -> Path:
        """Not relevant for html output"""
        fname = Path(fname)
        self.save(fname.with_suffix("." + self.file_ending))
        return fname


class LatexReport(Report):
    """Latex code generator"""

    block_start = r"""\documentclass[12pt, a4paper]{report}
\usepackage[top=3cm, bottom=3cm, left = 2cm, right = 2cm]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{float}
\usepackage{caption}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{amsmath} % red arrow for listing line break
\usepackage[hidelinks]{hyperref} % for links within the document

\lstset{
breaklines=true,
postbreak=\mbox{{$\hookrightarrow$}\space},
}


\begin{document}

\tableofcontents

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

    @property
    def file_ending(self) -> str:
        """file ending for generated files"""
        return "tex"

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

    def compile(self, fname: str | Path) -> Path:
        """Compile report using pdflatex
        If not present, the required .pdf and .tex suffixes are added as needed.

        returns the path of the generated pdf file
        """
        fname = Path(fname)
        self.save(fname.with_suffix("." + self.file_ending))

        # run twice so that the table of contents is generated correctly
        for _ in range(2):
            retval = subprocess.run(
                ["pdflatex", "-halt-on-error", fname.with_suffix(".tex").name],
                cwd=fname.parent,
                check=False,
                capture_output=True,
            )
            if retval.returncode != 0:
                print(retval.stdout.decode())
                print()
                print(retval.stderr.decode())
                raise RuntimeError("PDF generation with pdflatex failed.")

        return fname.with_suffix(".pdf")


class HTMLReport(Report):
    """HTML code generator"""

    block_start = r"""<html>
<head>
  <style>
  html * {
    font-size: 1vw;
  }

  </style>
</head>

<body style="margin:1vw;padding:1vw">

"""

    block_end = "</body>\n</html>"

    @property
    def file_ending(self) -> str:
        """file ending for generated files"""
        return "html"

    @staticmethod
    def _generate_entry(entry):
        """Generate latex code for the given entry"""
        if isinstance(entry, str):
            return "<p>" + entry.replace("\n", "<br/>\n") + "</p>\n"
        if isinstance(entry, ReportElement):
            return entry.html() + "\n"
        raise ValueError(
            f"Entry has unexpected type {type(entry)}. Only str, and ReportEntry are allowed."
        )

    def generate(self, structure: dict | None = None, level: int = 0) -> str:
        """Generate latex code"""
        if level >= 6:
            raise ValueError("Maximum sectioning depth reached.")

        if structure is None:
            structure = self

        html = ""
        for name, entry in structure.items():
            html += f"<h{level+1}>{name}</h{level+1}>\n"

            if isinstance(entry, dict):
                html += self.generate(entry, level + 1)
            elif isinstance(entry, list):
                for subentry in entry:
                    html += self._generate_entry(subentry)
            else:
                html += self._generate_entry(entry)

        if level == 0:
            html = self.block_start + html + self.block_end
        return html
