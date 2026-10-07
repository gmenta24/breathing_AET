# Code for `paper/main.tex`

| file | what |
|---|---|
| `aet_wdm_analyses.ipynb` | the three analyses (executed, with outputs); figures go to `../paper/figures/` |
| `aet_wdm_analyses.py` | percent-format source of the notebook (`python make_notebook.py` regenerates the `.ipynb`) |
| `aet_wdm.py` | noise model, closed-form TDI-2 covariance, pytdi simulation, FD/WDM likelihoods, jexplore driver |
| `mc_bias_check.py` | Monte-Carlo of Analysis 1 over independent realisations (`python mc_bias_check.py N SEED0 OUT`) |
| `fill_paper_numbers.py` | writes the results table / dense-FD timings of the paper from `results/summary.json` |

**Environment.** `.venv` is a virtualenv layered on the conda env `globalfit`
(`--system-site-packages`) that only adds `pytdi==2.2.1`. A Jupyter kernel `aet-wdm` is installed
inside it; in VS Code select the interpreter `code/.venv/bin/python`, or register the kernel
for all frontends with `.venv/bin/python -m ipykernel install --user --name aet-wdm`.

**Running.** `.venv/bin/python -m nbconvert --to notebook --execute --inplace aet_wdm_analyses.ipynb`
takes about 7 minutes with the cached simulations in `data/` (set `REGENERATE = True` in the
notebook, or delete `data/*.npz`, to redraw the noise: about 5 more minutes).
