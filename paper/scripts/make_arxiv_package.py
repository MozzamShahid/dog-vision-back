"""Assemble an arXiv-ready submission tarball for the IEEE-format paper.

arXiv compiles the sources itself with pdfLaTeX and does NOT reliably run
BibTeX, so the generated .bbl is shipped instead of references.bib. Figures are
copied in as real files (the working tree uses a symlink, which a tarball
cannot carry).

Run from repo root, after build_ieee.py:
    python paper/scripts/make_arxiv_package.py
Output: paper/arxiv_submission.tar.gz
"""

import re
import shutil
import subprocess
import tarfile
from pathlib import Path

IEEE = Path("paper/ieee")
STAGE = Path("paper/arxiv")
TARBALL = Path("paper/arxiv_submission.tar.gz")
TEXNAME = "dog_vision_ieee.tex"


def main():
    if STAGE.exists():
        shutil.rmtree(STAGE)
    (STAGE / "figures").mkdir(parents=True)

    # 1. produce the .bbl by running tectonic with intermediates kept
    subprocess.run(["tectonic", "--keep-intermediates", TEXNAME],
                   cwd=IEEE, capture_output=True, text=True)
    bbl = IEEE / "dog_vision_ieee.bbl"
    if not bbl.exists():
        raise SystemExit("no .bbl produced — run paper/scripts/build_ieee.py first")

    tex = (IEEE / TEXNAME).read_text()

    # 2. only the figures actually referenced need shipping
    used = set(re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", tex))
    src_figs = Path("paper/figures")
    for name in sorted(used):
        shutil.copy(src_figs / name, STAGE / "figures" / name)

    shutil.copy(IEEE / TEXNAME, STAGE / TEXNAME)
    shutil.copy(bbl, STAGE / "dog_vision_ieee.bbl")
    # Ship the .bib as well: arXiv prefers the .bbl, but any toolchain that
    # re-runs BibTeX (tectonic does) would otherwise overwrite the good .bbl
    # with a broken one when the database is absent.
    shutil.copy(Path("paper/references.bib"), STAGE / "references.bib")

    with tarfile.open(TARBALL, "w:gz") as tar:
        for f in sorted(STAGE.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=str(f.relative_to(STAGE)))

    print(f"figures included: {len(used)}")
    for n in sorted(used):
        print("   ", n)
    print("tarball:", TARBALL, f"({TARBALL.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
