REPORT=project-description

.SUFFIXES:
.SUFFIXES: .bib .pdf .tex
.PHONY: clean

all: $(REPORT).pdf

$(REPORT).pdf: $(REPORT).tex $(REPORT).aux
	pdflatex $(REPORT).tex -draftmode
	pdflatex $(REPORT).tex

$(REPORT).bbl: $(REPORT).aux
	bibtex $(REPORT).aux

$(REPORT).aux: # $(REPORT).bib
	pdflatex $(REPORT).tex -draftmode
	pdflatex $(REPORT).tex -draftmode

install:
	install $(REPORT).pdf $(prefix)/

clean:
	rm -rf *.aux *.lof *.log *.lot *.toc *.bbl *.blg *.pdf *.synctex.gz
