"""Build the IEEE (IEEEtran) conference-format PDF from paper/outline.md.

Pipeline:
  outline.md
    -> split front matter / abstract / keywords / body
    -> strip manual section numbers (IEEEtran numbers sections itself)
    -> rewrite section cross-references (§4.2 -> Section IV-B)
    -> figures get \\label, in-text "Figure N" becomes \\ref (auto-numbering)
    -> pandoc --natbib  ->  LaTeX body
    -> longtable -> table* , figure -> figure*   (two-column safe)
    -> wrap in IEEEtran preamble
    -> tectonic (runs BibTeX automatically)

Run from repo root:  python paper/scripts/build_ieee.py
Output:              paper/ieee/dog_vision_ieee.pdf
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("paper")
SRC = ROOT / "outline.md"
OUT = ROOT / "ieee"
TEX = OUT / "dog_vision_ieee.tex"

TITLE = ("A Two-Stage Deep Learning Pipeline for Real-Time\\\\"
         "Multi-Breed Canine Identification")

AUTHOR = r"""\author{\IEEEauthorblockN{Mozzam Shahid}
\IEEEauthorblockA{\textit{BS Information Technology}\\
\textit{University of Education}\\
Lahore, Pakistan \\
mozzamshahid906@gmail.com}
}"""

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI", 7: "VII", 8: "VIII"}
ALPHA = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "H", 9: "I"}

# in-text "Figure N" -> label key (label derives from the image filename)
FIGKEY = {
    1: "fig1_architecture",
    2: "fig2_confusion_matrix_top20",
    4: "fig4_reliability_diagram",
    6: "fig6_sample_detections",
    8: "fig8_per_breed_accuracy",
}


def md_to_tex(fragment):
    """Convert a markdown fragment to LaTeX.

    Essential for the abstract: a raw '94.04%' would start a LaTeX comment and
    silently swallow the rest of the line, and '**bold**' would render literally.
    """
    out = subprocess.run(["pandoc", "-f", "markdown", "-t", "latex", "--wrap=preserve"],
                         input=fragment, capture_output=True, text=True, check=True).stdout
    return out.strip()


def split_document(md):
    """Return (abstract, keywords, body_markdown)."""
    m = re.search(r"^## Abstract\s*\n+(.+?)\n+\*\*Keywords:\*\*\s*(.+?)\n", md, re.S | re.M)
    if not m:
        sys.exit("could not locate Abstract / Keywords")
    abstract, keywords = m.group(1).strip(), m.group(2).strip()
    body_start = md.index("## 1. Introduction")
    return abstract, keywords, md[body_start:]


def strip_section_numbers(body):
    body = re.sub(r"^## \d+\.\s+", "## ", body, flags=re.M)
    body = re.sub(r"^### \d+\.\d+\s+", "### ", body, flags=re.M)
    return body


def rewrite_crossrefs(text):
    """§4.2 -> Section IV-B ; §4 -> Section IV ; 'Section 3' -> 'Section III'."""
    def sec_sub(m):
        sec = int(m.group(1))
        sub = m.group(2)
        if sub:
            return f"SectionZZNBSPZZ{ROMAN[sec]}-{ALPHA[int(sub)]}"
        return f"SectionZZNBSPZZ{ROMAN[sec]}"

    text = re.sub(r"§(\d+)(?:\.(\d+))?", sec_sub, text)
    # "Sections IV-D–IV-E" style ranges already handled above per-token
    text = re.sub(r"\bSection (\d+)\b", lambda m: f"SectionZZNBSPZZ{ROMAN[int(m.group(1))]}", text)
    return text


def rewrite_figures(body):
    """Strip 'Figure N:' caption prefixes, add labels, convert in-text refs."""
    # in-text references first (before captions lose their numbers)
    def ref_sub(m):
        n = int(m.group(1))
        key = FIGKEY.get(n)
        return f"Fig.ZZNBSPZZ\\ref{{fig:{key}}}" if key else m.group(0)

    # only plain prose mentions, never the caption lines (those start with '![')
    lines = body.split("\n")
    for i, ln in enumerate(lines):
        if not ln.startswith("!["):
            lines[i] = re.sub(r"Figure (\d+)", ref_sub, ln)
    body = "\n".join(lines)

    # captions: '![Figure N: text](figures/x.png)' -> '![text](figures/x.png){#fig:x}'
    def cap_sub(m):
        caption, path = m.group(1), m.group(2)
        caption = re.sub(r"^Figure \d+:\s*", "", caption)
        key = Path(path).stem
        return f"![{caption}]({path}){{#fig:{key}}}"

    body = re.sub(r"!\[([^\]]+)\]\((figures/[^)]+)\)", cap_sub, body)
    return body


def longtable_to_table(tex):
    """Convert pandoc longtables into full-width table* floats (twocolumn safe)."""
    # pandoc wraps each longtable in '{\def\LTcaptype{none} ... }' — drop the
    # opening line here and the dangling '}' after the conversion below.
    tex = re.sub(r"\{\\def\\LTcaptype\{none\}[^\n]*\n", "", tex)

    # The column spec may span several lines and contains nested braces
    # (e.g. '>{\raggedright}p{(\columnwidth - 8\tabcolsep) * \real{0.2}}'),
    # so scan for the balanced closing brace rather than regex-matching it.
    OPEN = r"\begin{longtable}[]{"
    while True:
        i = tex.find(OPEN)
        if i == -1:
            break
        j = i + len(OPEN)
        depth = 1
        while j < len(tex) and depth:
            if tex[j] == "{":
                depth += 1
            elif tex[j] == "}":
                depth -= 1
            j += 1
        colspec = tex[i + len(OPEN): j - 1]
        end = tex.find(r"\end{longtable}", j)
        inner = tex[j:end]
        tex = tex[:i] + _rebuild_table(colspec, inner) + tex[end + len(r"\end{longtable}"):]

    # remove the now-orphaned closing brace of the LTcaptype wrapper
    tex = re.sub(r"(\\end\{table\*\})\n\}", r"\1", tex)
    return tex


def _rebuild_table(colspec, inner):
    """Turn one longtable body into a full-width table* float."""
    cap = ""
    cm = re.search(r"\\caption\{(.*?)\}\\tabularnewline", inner, re.S)
    if cm:
        cap = cm.group(1)
    # body rows: everything after the last \endlastfoot / \endhead
    for marker in (r"\endlastfoot", r"\endhead"):
        if marker in inner:
            inner_body = inner.rsplit(marker, 1)[1]
            break
    else:
        inner_body = inner
    # header: first \toprule ... \midrule chunk
    hm = re.search(r"\\toprule\\noalign\{\}\s*(.*?)\\midrule\\noalign\{\}", inner, re.S)
    header = hm.group(1).strip() if hm else ""
    rows = inner_body.replace(r"\bottomrule\noalign{}", "").strip()
    # rows already terminate with '\\'; adding another yields an empty
    # trailing row and a '\@array doesn't match its definition' error.
    parts = [
        r"\begin{table*}[t]", r"\centering",
        (r"\caption{%s}" % cap) if cap else "",
        r"\begin{tabular}{%s}" % colspec,
        r"\toprule", header, r"\midrule", rows, r"\bottomrule",
        r"\end{tabular}", r"\end{table*}",
    ]
    return "\n".join(p for p in parts if p)


def attach_table_captions(tex):
    """Fold '\\emph{Table (tab:key): text}' paragraphs into the following float.

    In markdown the caption is an italic paragraph preceding the table; left
    alone it stays in the body text while the table floats to a page top, so the
    table ends up uncaptioned. Move it inside as a real \\caption + \\label so
    IEEEtran numbers it and \\ref works.
    """
    pat = re.compile(
        r"\\emph\{Table \((tab:[a-z]+)\):\s*(.*?)\}\s*\n\s*\n(\\begin\{table\*\}\[t\]\n\\centering\n)",
        re.S)

    def repl(m):
        key, caption, opening = m.group(1), m.group(2).strip(), m.group(3)
        caption = " ".join(caption.split())
        return f"{opening}\\caption{{{caption}}}\n\\label{{{key}}}\n"

    tex, n = pat.subn(repl, tex)
    print(f"table captions attached: {n}")

    # A '†' note written after a table in markdown would otherwise be orphaned
    # in the body once the table floats; fold it into that table's caption.
    note = re.compile(r"(\\caption\{)(.*?)(\}\n\\label\{tab:[a-z]+\}\n)(.*?\\end\{table\*\})\n\n†\s*(.*?)\n\n",
                      re.S)

    def note_repl(m):
        body_note = " ".join(m.group(5).split())
        return (f"{m.group(1)}{m.group(2)} \\textsuperscript{{\\dag}}{body_note}"
                f"{m.group(3)}{m.group(4)}\n\n")

    tex, k = note.subn(note_repl, tex)
    print(f"table footnotes folded into captions: {k}")
    return tex


def widen_figures(tex):
    tex = tex.replace(r"\begin{figure}", r"\begin{figure*}[t]")
    tex = tex.replace(r"\end{figure}", r"\end{figure*}")
    tex = re.sub(r"\\includegraphics(\[[^\]]*\])?\{",
                 r"\\includegraphics[width=\\textwidth]{", tex)
    return tex


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = SRC.read_text()
    abstract, keywords, body = split_document(md)

    body = strip_section_numbers(body)
    body = rewrite_crossrefs(body)
    body = rewrite_figures(body)
    abstract = md_to_tex(rewrite_crossrefs(abstract)).replace("ZZNBSPZZ", "~")
    keywords = md_to_tex(keywords).replace("ZZNBSPZZ", "~")

    tmp_md = OUT / "_body.md"
    tmp_md.write_text(body)

    tex_body = subprocess.run(
        ["pandoc", str(tmp_md), "-t", "latex", "--natbib", "--wrap=preserve",
         "--shift-heading-level-by=-1"],
        capture_output=True, text=True, check=True).stdout

    # IEEEtran uses plain \cite; natbib's \citep/\citet are undefined here.
    tex_body = re.sub(r"\\cite[pt]\b", r"\\cite", tex_body)
    # restore intentional non-breaking spaces (pandoc escapes a literal '~')
    tex_body = tex_body.replace("ZZNBSPZZ", "~")
    # \pandocbounded is defined only in pandoc's own template
    tex_body = re.sub(r"\\pandocbounded\{(\\includegraphics[^}]*\{[^}]*\})\}", r"\1", tex_body)

    tex_body = longtable_to_table(tex_body)
    tex_body = attach_table_captions(tex_body)
    tex_body = widen_figures(tex_body)
    # pandoc renders '#fig:x' ids as \label already; ensure hypertarget noise is gone
    tex_body = re.sub(r"\\hypertarget\{[^}]*\}\{%\n", "", tex_body)
    tex_body = tex_body.replace("\\label{fig:", "\\label{fig:")

    doc = rf"""\documentclass[conference]{{IEEEtran}}
\IEEEoverridecommandlockouts
\usepackage{{cite}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{longtable}}
\usepackage{{url}}
\usepackage{{textcomp}}
% Engine-agnostic so this same source builds locally under tectonic/XeTeX and
% on arXiv under pdfLaTeX. newtx supplies the Times faces IEEE expects in both.
\usepackage{{iftex}}
\ifPDFTeX
  \usepackage[T1]{{fontenc}}
  \usepackage[utf8]{{inputenc}}
\fi
\usepackage{{newtxtext,newtxmath}}
\usepackage{{newunicodechar}}
\newunicodechar{{≈}}{{\ensuremath{{\approx}}}}
\newunicodechar{{≤}}{{\ensuremath{{\leq}}}}
\newunicodechar{{≥}}{{\ensuremath{{\geq}}}}
\newunicodechar{{∈}}{{\ensuremath{{\in}}}}
\newunicodechar{{→}}{{\ensuremath{{\rightarrow}}}}
\newunicodechar{{←}}{{\ensuremath{{\leftarrow}}}}
\newunicodechar{{×}}{{\ensuremath{{\times}}}}
\newunicodechar{{α}}{{\ensuremath{{\alpha}}}}
\newunicodechar{{σ}}{{\ensuremath{{\sigma}}}}
\newunicodechar{{θ}}{{\ensuremath{{\theta}}}}
\newunicodechar{{τ}}{{\ensuremath{{\tau}}}}
\newunicodechar{{μ}}{{\ensuremath{{\mu}}}}
\newunicodechar{{λ}}{{\ensuremath{{\lambda}}}}
\newunicodechar{{β}}{{\ensuremath{{\beta}}}}
\newunicodechar{{γ}}{{\ensuremath{{\gamma}}}}
\newunicodechar{{η}}{{\ensuremath{{\eta}}}}
\newunicodechar{{Δ}}{{\ensuremath{{\Delta}}}}
\newunicodechar{{ℓ}}{{\ensuremath{{\ell}}}}
\newunicodechar{{ℝ}}{{\ensuremath{{\mathbb{{R}}}}}}
\newunicodechar{{₁}}{{\ensuremath{{_1}}}}
\newunicodechar{{₂}}{{\ensuremath{{_2}}}}
\newunicodechar{{ₙ}}{{\ensuremath{{_n}}}}
\newunicodechar{{−}}{{-}}
\providecommand{{\tightlist}}{{\setlength{{\itemsep}}{{0pt}}\setlength{{\parskip}}{{0pt}}}}
% pandoc >=3.1.7 wraps images in \pandocbounded, which lives only in pandoc's
% own template; widths are set explicitly below so a passthrough is correct.
\providecommand{{\pandocbounded}}[1]{{#1}}
% pandoc computes proportional column widths as '(\columnwidth - N\tabcolsep) * \real{{0.2}}'
\usepackage{{calc}}
\usepackage{{array}}
\providecommand{{\real}}[1]{{#1}}

\begin{{document}}

\title{{{TITLE}}}
{AUTHOR}

\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

\begin{{IEEEkeywords}}
{keywords}
\end{{IEEEkeywords}}

{tex_body}

\bibliographystyle{{IEEEtran}}
\bibliography{{references}}

\end{{document}}
"""
    TEX.write_text(doc)
    shutil.copy(ROOT / "references.bib", OUT / "references.bib")
    if not (OUT / "figures").exists():
        (OUT / "figures").symlink_to(Path("..") / "figures")

    print("wrote", TEX)
    r = subprocess.run(["tectonic", TEX.name], cwd=OUT, capture_output=True, text=True)
    tail = (r.stderr or "").strip().splitlines()[-12:]
    print("\n".join(tail))
    pdf = OUT / "dog_vision_ieee.pdf"
    print("PDF:", pdf, "exists:", pdf.exists())


if __name__ == "__main__":
    main()
