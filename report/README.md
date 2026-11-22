# CS5478 Final Report

This directory contains an Overleaf-compatible first draft of the CS5478
course report. It uses the unmodified official NeurIPS 2026 style file with the
`preprint` option; this is a course report, not a NeurIPS submission.

Official sources:

- Template ZIP: <https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip>
- Call for Papers: <https://neurips.cc/Conferences/2026/CallForPapers>
- Retrieved: 2026-11-22
- `neurips_2026.sty` SHA-256:
  `c3fc2894e83d2517ca18b66741d6c595986d97957dc08ec08bb2125a7ec4555a`

## Build

```bash
cd report
conda run -n cs5478 tectonic -X compile main.tex \
  --outdir build --keep-logs --keep-intermediates
cp build/main.pdf CS5478_Final_Report.pdf
```

The local shell does not provide `latexmk`, `pdflatex`, or `xelatex`. The
standalone Tectonic 0.17.0 engine is installed in the `cs5478` Conda
environment and compiles the report, including BibTeX, without a full TeX
distribution. The repository's VS Code settings invoke Tectonic through the
named Conda environment instead of relying on LaTeX Workshop's default
`latexmk` recipe.

## Repository

The public project repository is
<https://github.com/Xiaoda-Zhong/cs5478-difficulty-curriculum-bc>.
