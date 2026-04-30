# Pi-GRPO paper

A NeurIPS-style preprint covering the Pi-GRPO physics-informed RL stack. Self-contained: no external `.bib`, no PNG figures (TikZ only), so the source compiles on stock TeX Live and is ready for arXiv submission.

## Build

```bash
make pdf          # produces pi_grpo_neurips.pdf
make lint         # chktex if installed
make clean        # remove .aux/.log/.out
make distclean    # also remove the PDF
make arxiv        # build arxiv-build/ and a single-file tarball
```

## arXiv submission checklist

1. `make pdf` produces a clean PDF with no `Undefined references` or `Citation undefined` warnings.
2. `pdffonts pi_grpo_neurips.pdf` shows only Type 1 / TrueType fonts.
3. `make arxiv` produces `pi_grpo_neurips-arxiv.tar.gz`. Upload at <https://arxiv.org/submit>.
4. arXiv categories (suggested): `cs.LG`, `cs.AI`, `stat.ML`.
5. Companion preprint cross-link: cite the GeoTrace-Agent preprint once an arXiv ID is assigned.

## Citation

Once the arXiv ID is assigned, update the BibTeX entry in [`../CITATION.cff`](../CITATION.cff) and the `arXiv:xxxx.xxxxx` badge in [`../README.md`](../README.md).
