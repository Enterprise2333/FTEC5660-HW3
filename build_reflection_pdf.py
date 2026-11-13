"""Render the Task 2 reflection as a plain PDF without page decoration."""
from html import escape
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "REFLECTION.md"
OUTPUT = ROOT / "output" / "pdf" / "HW3_Reflection.pdf"


def main():
    sections = SOURCE.read_text(encoding="utf-8").strip().split("\n\n")
    title = sections[0].removeprefix("# ")
    body = sections[1:]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    title_style = ParagraphStyle(
        "Title", fontName="Helvetica-Bold", fontSize=15, leading=20, spaceAfter=18,
    )
    body_style = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=10.5, leading=15.5, spaceAfter=14,
    )
    story = [Paragraph(escape(title), title_style)]
    story += [Paragraph(escape(paragraph.replace("\n", " ")), body_style) for paragraph in body]
    document = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=23 * mm, rightMargin=23 * mm,
        topMargin=22 * mm, bottomMargin=22 * mm,
        title=title, author="Enterprise2333",
    )
    document.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    main()
