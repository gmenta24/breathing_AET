# Code for `paper/main.tex`

| file | what |
|---|---|
| `aet_wdm_analyses.ipynb` | the three analyses (executed, with outputs); figures go to `../paper/figures/` |
| `aet_wdm_analyses.py` | percent-format source of the notebook (`python make_notebook.py` regenerates the `.ipynb`) |
| `aet_wdm.py` | noise model, closed-form TDI-2 covariance, pytdi simulation, FD/WDM likelihoods, jexplore driver |
| `mc_bias_check.py` | Monte-Carlo of Analysis 1 over independent realisations (`python mc_bias_check.py N SEED0 OUT`) |
| `fill_paper_numbers.py` | writes the results table / dense-FD timings of the paper from `results/summary.json` |
| `diagonal_likelihood.ipynb` | WDM likelihood in diagonal form (branch `diagonal-likelihood`): checks of the identity `K_tm = 4 K_oms - 32 Psi^2` and of the closed-form eigenvalues against pytdi, 3x3 vs diagonal likelihood on cells and on every pixel (values, cost, posteriors); results in `results/diagonal_likelihood.json`, chains in `results/diag_chains.npz`, figures `results/diag_*.pdf` (not inline) |
| `diagonal_likelihood.py` | percent-format source of that notebook (`python make_notebook.py diagonal_likelihood.py diagonal_likelihood.ipynb`) |

**Diagonal form.** For any frozen delays the TM and OMS geometric factors of the closed form satisfy
`G_tm = 4 G_oms - 32 diag(Psi^2)` with `Psi_X = |G_X| sin(w L2) sin(w L3)` (and cyclically), so one basis per
frequency (per WDM pixel), independent of `A` and `P`, diagonalises the covariance:
`v_j^T C v_k = delta_jk (P^2 S_oms + kappa_j A^2 S_tm)`.  In `aet_wdm.py`: `tdi2_delta`, `analytic_basis`
(closed-form basis and `kappa_j`), `kappa_closed_form` (the cubic of the paper), `sym3_eigh` (closed-form 3x3
eigen-decomposition), `simdiag` (exact basis of window-averaged pixel/cell covariances), `project_diag`,
`make_loglik_eig` (diagonal likelihood) and `make_loglik_full_lapack` (LAPACK reference, for timing).

**Environment.** `.venv` is a virtualenv layered on the conda env `globalfit`
(`--system-site-packages`) that only adds `pytdi==2.2.1`. A Jupyter kernel `aet-wdm` is installed
inside it; in VS Code select the interpreter `code/.venv/bin/python`, or register the kernel
for all frontends with `.venv/bin/python -m ipykernel install --user --name aet-wdm`.

**Running.** `.venv/bin/python -m nbconvert --to notebook --execute --inplace aet_wdm_analyses.ipynb`
takes about 7 minutes with the cached simulations in `data/` (set `REGENERATE = True` in the
notebook, or delete `data/*.npz`, to redraw the noise: about 5 more minutes).

**Diagonal vs 3x3 (ESA orbits, 32 walkers, 1 557 528 pixels per channel, 22-core CPU).** One evaluation: 3x3
cofactors ~139 ms, diagonal form ~81 ms (with one logarithm of the product of the three eigenvalues; three
logarithms take ~313 ms), 3x3 LAPACK 176 s; value and gradient 6.9 s vs 0.32 s; MCMC 790 s vs 330 s.  Posteriors:
identical with the exact pixel basis, within 0.03 sigma with the closed-form basis; taking the eigenvalues at the
band centre instead of window-averaging the projected covariance biases `A` by -3.7 sigma.
