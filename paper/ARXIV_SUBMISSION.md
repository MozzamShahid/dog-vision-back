# arXiv Submission Sheet

Everything needed to submit the paper. Upload **`paper/arxiv_submission.tar.gz`** — the
LaTeX source. Do **not** upload a PDF; arXiv compiles the source itself.

Regenerate the tarball any time with:

```bash
export PATH="/opt/homebrew/bin:$PATH"
./venv/bin/python paper/scripts/build_ieee.py
./venv/bin/python paper/scripts/make_arxiv_package.py
```

---

## 0. Before you start: endorsement

arXiv requires **endorsement** for a first submission to a category you have not
published in. This is the most common thing that blocks a new author.

- If you register with an email from a recognised academic institution, you may be
  auto-endorsed.
- Otherwise arXiv shows an endorsement code and you must ask someone who has already
  published in `cs.CV` to endorse you. A former supervisor or any published researcher
  you know can do it in about two minutes.
- Endorsement is about identity/relevance, not peer review — it is not a quality judgement.

Register first at <https://arxiv.org/user/register>; arXiv tells you whether endorsement
is needed when you start the submission.

---

## 1. Metadata to paste into the form

**Title**

```
A Two-Stage Deep Learning Pipeline for Real-Time Multi-Breed Canine Identification
```

**Authors**

```
Mozzam Shahid
```

**Abstract** — paste from `paper/arxiv_abstract.txt` (1,908 chars; arXiv's limit is 1,920).
It is plain text with no LaTeX or markdown markup, which is what the field expects.

**Primary category**

```
cs.CV  (Computer Vision and Pattern Recognition)
```

**Cross-list (optional, recommended)**

```
cs.LG  (Machine Learning)
```

**Comments field**

```
18 pages, 11 figures, 9 tables. IEEE conference format
```

**License** — `CC BY 4.0` is the usual choice for a preprint you want widely read and
cited. `arXiv non-exclusive license` is the most conservative option if you would rather
not permit redistribution. Either is fine; CC BY 4.0 is recommended.

---

## 2. Submission steps

1. Log in, then **Start New Submission**.
2. Choose license and primary category (`cs.CV`).
3. Upload `arxiv_submission.tar.gz`.
4. arXiv compiles it and shows a generated PDF. **Read that PDF carefully** — it is what
   gets published, not the local build. Check the figures appear, tables are numbered
   TABLE I–IX, and the references resolve to numbers rather than question marks.
5. Paste the title, authors, abstract, comments.
6. Preview, then **Submit**.

Announcement is typically the next business day. You get an arXiv ID (e.g.
`arXiv:2607.XXXXX`) — put that in the repo README once it is live.

---

## 3. What is in the tarball

| File | Purpose |
|------|---------|
| `dog_vision_ieee.tex` | The paper. Engine-agnostic preamble so arXiv's pdfLaTeX and a local XeTeX build both work. |
| `dog_vision_ieee.bbl` | Pre-built bibliography — arXiv does not reliably run BibTeX. |
| `references.bib` | Shipped as well so any toolchain that *does* re-run BibTeX succeeds. |
| `figures/*.png` | The 11 figures actually referenced by the paper. |

Verified: extracting the tarball into an empty directory and compiling produces an
18-page PDF with 0 unresolved references and all 53 bibliography entries.

---

## 4. Known caveats to be aware of before you publish

- **Evaluation protocol.** Reported accuracy is computed over the full 20,580-image
  Stanford Dogs Dataset rather than a disjoint held-out split. This is stated openly as
  Limitation 7, and no table claims a held-out number. If a reader raises it, the clean
  held-out figure can be produced with
  `python paper/scripts/exp1_validation_suite.py --val-only`.
- **Three citation years were corrected** to match the actual publications: Gunter et al.
  2018 (not 2021), Beery et al. 2018 (not 2021), Nguyen et al. 2017. Worth confirming
  these are the works you intended to cite.
- **Figure 10 (training curves) is absent** — no per-epoch training log was saved. It is
  listed under Future Work.
- arXiv submissions can be **replaced** with a new version at any time, so none of the
  above has to block posting v1.
