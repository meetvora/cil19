LATEXMK = latexmk
RM = rm -f

BASENAME = main
TEXFILES = $(wildcard *.tex)
BIBFILES = $(wildcard *.bib)
IMGFILES = $(wildcard *.png)
SRCS = $(TEXFILES) $(BIBFILES) $(IMGFILES)

all: $(BASENAME).pdf

$(BASENAME).pdf: $(SRCS)
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -bibtex \
		-use-make $(BASENAME).tex

clean:
	latexmk -bibtex -CA

