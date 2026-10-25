"""REPORT.md -> report.html -> L230518_VisionGuard.pdf (headless Chrome, no pandoc needed).

    pip install markdown
    python report/build_pdf.py
"""
import re, subprocess, sys
from pathlib import Path
import markdown

HERE = Path(__file__).resolve().parent
CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe",
          r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 16mm 15mm; }
body { font: 10.5pt/1.45 "Segoe UI", Calibri, sans-serif; color: #1d2330; }
h1 { font-size: 17pt; border-bottom: 2px solid #3b5bdb; padding-bottom: 4px; margin-top: 26px; page-break-before: always; }
h1.title { page-break-before: avoid; border: 0; font-size: 22pt; text-align: center; margin-top: 60px; }
p.subtitle { text-align: center; color: #555; font-size: 13pt; }
h2 { font-size: 13pt; color: #3b5bdb; margin-top: 18px; }
table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccd; padding: 3px 7px; vertical-align: top; }
th { background: #eef1f8; }
code { font: 8.8pt Consolas, monospace; background: #f3f4f6; padding: 0 3px; }
pre { background: #f3f4f6; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
figure { margin: 10px 0; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 120mm; border: 1px solid #ddd; }
figcaption { font-size: 9pt; color: #444; font-style: italic; margin-top: 3px; }
hr { display: none; }
"""

def main():
    src = (HERE / "REPORT.md").read_text(encoding="utf-8")
    meta = re.match(r"---\n(.*?)\n---\n", src, re.S)
    title = sub = ""
    if meta:
        for line in meta.group(1).splitlines():
            k, _, v = line.partition(":")
            if k.strip() == "title": title = v.strip().strip('"')
            if k.strip() == "subtitle": sub = v.strip().strip('"')
        src = src[meta.end():]
    html = markdown.markdown(src, extensions=["tables", "fenced_code"])
    # standalone images -> captioned figures
    html = re.sub(r'<p>(?:⏳ )?<img alt="([^"]*)" src="([^"]+)" ?/?>\s*</p>',
                  r'<figure><img src="\2"><figcaption>\1</figcaption></figure>', html)
    html = re.sub(r'<img alt="([^"]*)" src="([^"]+)" ?/?>',
                  r'<figure><img src="\2"><figcaption>\1</figcaption></figure>', html)
    head = f'<h1 class="title">{title}</h1><p class="subtitle">{sub}</p>' if title else ""
    out = HERE / "report.html"
    out.write_text(f'<!doctype html><meta charset="utf-8"><title>{title}</title><style>{CSS}</style>{head}{html}',
                   encoding="utf-8")
    chrome = next((c for c in CHROME if Path(c).exists()), None)
    if not chrome:
        sys.exit(f"wrote {out}; no Chrome/Edge found, print it to PDF manually")
    pdf = HERE / "L230518_VisionGuard.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={pdf}", out.as_uri()], check=True, capture_output=True)
    print("wrote", pdf)

if __name__ == "__main__":
    main()
