"""LaTeX: Utilities to export results in LaTeX.
"""
# Author: Borja Ayerdi <ayerdi.borja@gmail.com>
# License: BSD 3 clause
# Copyright: UPV/EHU


def image_to_latex(title,dirname):
    """
    Given an image title and directory return in LaTeX format.

    Parameters
    ----------
    title: String
        Image title.
    dirname: String
        Image directory.

    Returns
    -------
    String
        LaTeX text result.

    """
    return "\\begin{figure}[ht!]\n\
\\centering\n\
\\includegraphics[width=90mm]{" + dirname +"\n\
\\caption{"+title+" \\label{overflow}}\n\
\\end{figure}\n"
