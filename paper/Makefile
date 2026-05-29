PAPER := pi_grpo_neurips
TEX   := $(PAPER).tex
PDF   := $(PAPER).pdf

LATEX     ?= pdflatex
LATEXOPTS ?= -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: pdf clean lint distclean view arxiv

pdf: $(PDF)

$(PDF): $(TEX)
	$(LATEX) $(LATEXOPTS) $(TEX)
	$(LATEX) $(LATEXOPTS) $(TEX)

lint:
	@command -v chktex >/dev/null 2>&1 && chktex -q $(TEX) || echo "chktex not installed; skipping"

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(PDF)

view: $(PDF)
	@open $(PDF) || xdg-open $(PDF) || true

arxiv: $(PDF)
	@mkdir -p arxiv-build
	cp $(TEX) arxiv-build/
	cd arxiv-build && tar -czf ../$(PAPER)-arxiv.tar.gz $(TEX)
	@echo "arxiv tarball: $(PAPER)-arxiv.tar.gz"
