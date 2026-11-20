"""Assemble the final report from the per-section Markdown files.

Each teammate edits their own section in docs/report/. This script joins them,
in order, into docs/report/report.md: it drops the "(draft — name)" tags,
the "Slide bullets" and per-section "Credits" blocks, and shifts headings down
one level under a single report title.

Example:
    python scripts/build_report.py
"""

import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "docs", "report")
SECTIONS = ["introduction.md", "dataset_preprocessing.md", "model_training.md",
            "results.md", "conclusion.md"]
DROP_HEADINGS = {"slide bullets", "credits"}
TITLE = """# Crowd Counting and Density Estimation with CSRNet

**Team CrowdStrike:** Prashant Panta, Ronish Karki, Prajwal Kunwar

Code: https://github.com/Prajwal382/CrowdStrike
"""


def clean_section(text: str) -> str:
    out, skip_level = [], None
    for line in text.splitlines():
        m = re.match(r"^(#+)\s+(.*)$", line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            if skip_level is not None and level <= skip_level:
                skip_level = None
            if title.lower() in DROP_HEADINGS:
                skip_level = level
                continue
            title = re.sub(r"\s*\(draft\s*[—-][^)]*\)\s*$", "", title)
            line = "#" * (level + 1) + " " + title
        if skip_level is None:
            out.append(line)
    return "\n".join(out).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPORT_DIR, "report.md"),
                    help="output file (default: %(default)s)")
    args = ap.parse_args()

    parts = [TITLE]
    for name in SECTIONS:
        path = os.path.join(REPORT_DIR, name)
        if not os.path.exists(path):
            print(f"warning: missing {name}, skipped")
            continue
        with open(path, encoding="utf-8") as f:
            parts.append(clean_section(f.read()))
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(parts))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
