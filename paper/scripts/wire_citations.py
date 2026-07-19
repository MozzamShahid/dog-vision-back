"""Convert human-readable prose citations in outline.md to pandoc [@key] syntax.

Handles both styles used in the paper:
  bracket : "[Khosla et al., 2011]"           -> "[@khosla2011stanforddogs]"
            "[Guo et al., 2017; Fort et al., 2019]" -> "[@guo...; @fort...]"
  name    : "Hsu [2015] demonstrated"          -> "Hsu [@hsu2015dogclassification] demonstrated"

Markdown image/link syntax (![alt](path), [text](url)) is skipped so figure
captions are never mangled. Any citation that cannot be mapped is reported
rather than silently dropped.

Run: python paper/scripts/wire_citations.py [--write]
"""

import re
import sys
from pathlib import Path

SRC = Path("paper/outline.md")

# prose citation string -> bibtex key
CITES = {
    "Abdar et al., 2021": "abdar2021uncertaintyreview",
    "Bewley et al., 2016": "bewley2016sort",
    "Bochkovskiy et al., 2020": "bochkovskiy2020yolov4",
    "Collier, 2006": "collier2006bsl",
    "Dosovitskiy et al., 2021": "dosovitskiy2021vit",
    "Felzenszwalb et al., 2010": "felzenszwalb2010dpm",
    "Fette and Melnikov, 2011": "fette2011websocket",
    "Fort et al., 2019": "fort2019deepensembles",
    "Girshick et al., 2014": "girshick2014rcnn",
    "Girshick, 2015": "girshick2015fastrcnn",
    "Gunter et al., 2021": "gunter2018caneidentity",
    "Guo et al., 2017": "guo2017temperaturescaling",
    "He et al., 2017": "he2017maskrcnn",
    "Howard et al., 2017": "howard2017mobilenets",
    "Howard et al., 2019": "howard2019mobilenetv3",
    "Khosla et al., 2011": "khosla2011stanforddogs",
    "Lakshminarayanan et al., 2017": "lakshminarayanan2017ensembles",
    "Lin et al., 2017": "lin2017fpn",
    "Liu et al., 2016": "liu2016ssd",
    "Liu et al., 2021": "liu2021swin",
    "Liu et al., 2022": "liu2022convnext",
    "Loshchilov and Hutter, 2019": "loshchilov2019adamw",
    "Müller et al., 2020": "muller2020labelsmoothing",
    "Olston et al., 2017": "olston2017tensorflowserving",
    "Packer and Tivers, 2015": "packer2015brachycephalic",
    "Patronek et al., 2013": "patronek2013dogbite",
    "Redmon and Farhadi, 2017": "redmon2017yolo9000",
    "Redmon and Farhadi, 2018": "redmon2018yolov3",
    "Redmon et al., 2016": "redmon2016yolo",
    "Ren et al., 2015": "ren2015fasterrcnn",
    "Sandler et al., 2018": "sandler2018mobilenetv2",
    "Shankar et al., 2017": "shankar2017geodiversity",
    "Tan and Le, 2019": "tan2019efficientnet",
    "Tan and Le, 2021": "tan2021efficientnetv2",
    "Tan et al., 2020": "tan2020efficientdet",
    "Touvron et al., 2021": "touvron2021deit",
    "Weiss et al., 2012": "weiss2012adoption",
    "Wojke et al., 2017": "wojke2017deepsort",
    "Yun et al., 2019": "yun2019cutmix",
    "Zhang et al., 2018": "zhang2018mixup",
}

# "Author et al. [YYYY]" narrative style -> key. Year in prose may differ from
# the actual publication year; the key is authoritative.
NARRATIVE = {
    ("Ashukha et al.", "2020"): "ashukha2020pitfalls",
    ("Baylor et al.", "2017"): "baylor2017tfx",
    ("Beery et al.", "2021"): "beery2018terraincognita",
    ("Biørn-Hansen et al.", "2017"): "biornhansen2017pwa",
    ("Borwarnginn et al.", "2021"): "borwarnginn2021dogbreed",
    ("Crankshaw et al.", "2017"): "crankshaw2017clipper",
    ("Guo et al.", "2017"): "guo2017temperaturescaling",
    ("Hsu", "2015"): "hsu2015dogclassification",
    ("Khosla et al.", "2011"): "khosla2011stanforddogs",
    ("Lakshminarayanan et al.", "2017"): "lakshminarayanan2017ensembles",
    ("Nguyen et al.", "2017"): "nguyen2017animalrecognition",
    ("Norouzzadeh et al.", "2018"): "norouzzadeh2018cameratrap",
    ("Oluleye et al.", "2024"): "oluleye2024dogbreed",
    ("Raduly et al.", "2018"): "raduly2018dogrecognition",
    ("Wang et al.", "2022"): "wang2022dogidentification",
    ("Zou et al.", "2020"): "zou2020finegraineddog",
}

unmapped = []


def sub_narrative(text):
    names = sorted(NARRATIVE.items(), key=lambda kv: -len(kv[0][0]))
    for (name, year), key in names:
        pat = re.escape(name) + r"\s\[" + year + r"\]"
        text = re.sub(pat, f"{name} [@{key}]", text)
    return text


def sub_bracket(text):
    def repl(m):
        # skip markdown image alt / link text
        if m.start() > 0 and text[m.start() - 1] == "!":
            return m.group(0)
        end = m.end()
        if end < len(text) and text[end] == "(":
            return m.group(0)
        inner = m.group(1)
        if "@" in inner:            # already converted
            return m.group(0)
        parts = [p.strip() for p in inner.split(";")]
        keys = []
        for p in parts:
            if p in CITES:
                keys.append(CITES[p])
            else:
                return m.group(0)   # not a citation -> leave alone
        return "[" + "; ".join("@" + k for k in keys) + "]"

    return re.sub(r"\[([^\[\]]+)\]", repl, text)


def main():
    text = SRC.read_text()
    out = sub_narrative(text)
    out = sub_bracket(out)

    # report anything that still looks like an unconverted citation
    leftovers = set()
    for m in re.finditer(r"\[([^\[\]@]*\b(?:19|20)\d{2}\b[^\[\]@]*)\]", out):
        inner = m.group(1)
        nxt = out[m.end():m.end() + 1]
        if nxt == "(" or inner.startswith("Figure") or inner.startswith("Table"):
            continue
        leftovers.add(inner.strip())
    if leftovers:
        print("!! UNCONVERTED citation-like text:")
        for l in sorted(leftovers):
            print("   -", l)

    n = out.count("[@")
    print(f"citation groups written: {n}")
    if "--write" in sys.argv:
        SRC.write_text(out)
        print("wrote", SRC)
    else:
        print("(dry run; pass --write to apply)")


if __name__ == "__main__":
    main()
