Using the template
------------------

This is the TU Delft LaTeX template for reports and theses. It is designed to
work with all versions of LaTeX, but if you want to adhere to the TU Delft house
style, you need to use XeLaTeX, as it supports TrueType and OpenType fonts. The
document can be compiled with

  xelatex report
  bibtex report
  xelatex report
  xelatex report

This is equivalent to selecting 'XeLaTeX+BibTeX' or similar in your favorite TeX
editing program.

A sample document, as well as documentation for template options, can be found
in example.pdf. An example with the native LaTeX fonts, compiled using the
'nativefonts' option (or with pdflatex), can be found in
example-nativefonts.pdf.

A separate example document is available which generates a cover image (front,
back and spine). This document can be generated with

  xelatex cover
  xelatex cover

or simply with the 'XeLaTeX' option in TeXworks or an equivalent program.



Installation on Windows
-----------------------

The TU Delft LaTeX template has been tested to work with the most recent version
of MiKTeX at the time of this writing (2.9). The following packages are required
on top of a basic MiKTeX installation to make full use of the template:

  caption, fancyhdr, filehook, footmisc, fourier, l3kernel, l3packages,
  metalogo, mptopdf, ms, natbib, pgf, realscripts, tipa, titlesec, tocbibind,
  unicode-math, url, xcolor, xetex-def

Note that MiKTeX will generally automatically install these packages if they are
missing from your installation.



Installation on Linux (Debian/Ubuntu)
-------------------------------------

Recent versions of Debian, and derived distributions such as Ubuntu, use the TeX
Live system. Install the following packages to make full use of the this
template:

  texlive, texlive-fonts-extra, texlive-math-extra, texlive-lang-dutch,
  texlive-lang-english, texlive-latex-extra, texlive-xetex

