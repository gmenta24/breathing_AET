"""Convert the percent-format script ``aet_wdm_analyses.py`` into ``aet_wdm_analyses.ipynb``.

Cells start with ``# %%`` (code) or ``# %% [markdown]`` (markdown, lines prefixed by ``# ``).
Execute afterwards with
``.venv/bin/python -m nbconvert --to notebook --execute --inplace aet_wdm_analyses.ipynb``.
"""
import sys

import nbformat

src = sys.argv[1] if len(sys.argv) > 1 else "aet_wdm_analyses.py"
dst = sys.argv[2] if len(sys.argv) > 2 else "aet_wdm_analyses.ipynb"
cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if kind == "markdown":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    elif text.strip():
        cells.append(nbformat.v4.new_code_cell(text))


for line in open(src).read().splitlines():
    if line.startswith("# %%"):
        flush()
        kind, buf = ("markdown" if "[markdown]" in line else "code"), []
    else:
        buf.append(line)
flush()
nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "aet-wdm", "display_name": "Python (AET WDM venv)", "language": "python"}
nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, dst)
print(f"wrote {dst}: {len(cells)} cells")
