#!/usr/bin/env python
"""Render the coursework report Markdown to a self-contained, printable document.

The report (`report/coursework_report.md`) references real figures under
`outputs/figures/`. This script converts the Markdown to HTML, **inlines every
referenced image as a base64 data URI** (so the HTML is fully self-contained and
survives being moved or emailed), and applies a clean print stylesheet.

Output: ``report/coursework_report.html``. Open it in any browser and use
``Ctrl+P -> Save as PDF`` to produce the final PDF (figures included).

If the optional ``xhtml2pdf`` package is installed, a ``.pdf`` is also written
directly (``pip install xhtml2pdf``); otherwise the script prints the
print-to-PDF instruction.

Usage
-----
    python scripts/build_report.py
"""
from __future__ import annotations

import base64
import mimetypes
import re
from pathlib import Path

import markdown  # pip install markdown (already in requirements)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = PROJECT_ROOT / "report" / "coursework_report.md"
REPORT_HTML = PROJECT_ROOT / "report" / "coursework_report.html"
REPORT_PDF = PROJECT_ROOT / "report" / "coursework_report.pdf"

CSS = """
@page { size: A4; margin: 20mm 18mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 11pt; line-height: 1.5;
       color: #1a1a1a; max-width: 820px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 15pt; margin-top: 22px; border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 12.5pt; margin-top: 16px; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 10pt; }
th, td { border: 1px solid #bbb; padding: 5px 8px; text-align: left; }
th { background: #f0f0f0; }
code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 9.5pt; }
pre { background: #f6f8fa; border: 1px solid #ddd; border-radius: 5px; padding: 10px;
      overflow-x: auto; font-size: 9pt; line-height: 1.35; }
pre code { background: none; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 10px auto; border: 1px solid #eee; }
em { color: #444; }
blockquote { border-left: 3px solid #ccc; margin: 10px 0; padding: 4px 14px; color: #555; }
a { color: #0b5cad; text-decoration: none; }
"""


def _data_uri(img_path: Path) -> str | None:
    """Return a base64 data: URI for an image file, or None if it does not exist."""
    if not img_path.exists():
        print(f"  [warn] figure not found, skipping embed: {img_path}")
        return None
    mime = mimetypes.guess_type(img_path.name)[0] or "image/png"
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _inline_images(html: str) -> str:
    """Replace <img src="relative/path"> with embedded base64 data URIs.

    Image paths in the Markdown are relative to the report/ directory (e.g.
    ``../outputs/figures/foo.png``); they are resolved against REPORT_MD's parent.
    """
    def repl(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return match.group(0)
        resolved = (REPORT_MD.parent / src).resolve()
        uri = _data_uri(resolved)
        if uri is None:
            return match.group(0)
        return match.group(0).replace(src, uri)

    return re.sub(r'<img[^>]*\ssrc="([^"]+)"', repl, html)


AUTHOR = "Langhui Huang"


def _set_pdf_author(pdf_path: Path) -> None:
    """Rewrite the PDF's document info so it lists the author, not the tool.

    xhtml2pdf stamps ``Producer: xhtml2pdf`` into the metadata; this replaces the
    Author/Creator/Producer fields with the author's name. No-op if ``pypdf`` is
    not installed.
    """
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
    except Exception:
        return
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.add_metadata({"/Author": AUTHOR, "/Creator": AUTHOR, "/Producer": AUTHOR})
    tmp = pdf_path.with_suffix(".pdf.tmp")
    with open(tmp, "wb") as f:
        writer.write(f)
    tmp.replace(pdf_path)


def main() -> None:
    md_text = REPORT_MD.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])
    body = _inline_images(body)

    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>3D Scene Occupancy Completion — Coursework Report</title>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    REPORT_HTML.write_text(html, encoding="utf-8")
    print(f"Wrote self-contained HTML: {REPORT_HTML}")

    # Optional: emit a PDF directly if xhtml2pdf is available.
    try:
        from xhtml2pdf import pisa  # type: ignore
        with REPORT_PDF.open("wb") as f:
            result = pisa.CreatePDF(html, dest=f)
        if result.err:
            print("  [warn] xhtml2pdf reported errors; use the HTML + browser print instead.")
        else:
            _set_pdf_author(REPORT_PDF)
            print(f"Wrote PDF: {REPORT_PDF}")
    except Exception:
        print(
            "\nTo get a PDF: open the HTML above in a browser and use Ctrl+P -> 'Save as PDF'.\n"
            "(Optional one-step PDF: `pip install xhtml2pdf` then re-run this script.)"
        )


if __name__ == "__main__":
    main()
