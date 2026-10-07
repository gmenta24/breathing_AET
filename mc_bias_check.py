"""Monte-Carlo check of Analysis 1: over independent realisations, the FD and WDM
maximum-likelihood estimates must be unbiased (mean bias/sigma ~ 0, scatter ~ 1) and
must agree with each other realisation by realisation."""

import json
import os
import sys

import numpy as np

import aet_wdm as aw

NREAL = int(sys.argv[1]) if len(sys.argv) > 1 else 12
SEED0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
OUT = sys.argv[3] if len(sys.argv) > 3 else "results/mc_bias_check.json"
grid = aw.Grid(dt=10.0, nt=3072, nf=1024)
medges = aw.log_edges(2, 512, 100)
cells = aw.Cells(grid, medges, np.round(np.linspace(0, grid.nt, 49)).astype(int))
L_EQ = aw.L_NOMINAL / aw.C_LIGHT
fd_tm, fd_oms = aw.fd_model_equal_arm(cells, L_EQ)
wd_tm, wd_oms = aw.wdm_model_equal_arm(cells, L_EQ)
theta = np.array([aw.A_TRUE, aw.P_TRUE])

out = []
for r in range(NREAL):
    xyz = aw.simulate_tdi2(np.random.default_rng(SEED0 + r), grid.n, grid.dt, aw.A_TRUE, aw.P_TRUE,
                           np.full(6, L_EQ), periodic=True)
    I, K = aw.fd_periodograms(xyz, grid, cells)
    Y, D = aw.wdm_cells(aw.wdm_transform(xyz, grid), cells)
    zf = (np.real(np.einsum("cii->ci", I)) / (aw.A_TRUE**2 * fd_tm + aw.P_TRUE**2 * fd_oms) - 1) * np.sqrt(K)[:, None]
    zw = (np.einsum("tcii->tci", Y) / (aw.A_TRUE**2 * wd_tm + aw.P_TRUE**2 * wd_oms)[None] - 1) * np.sqrt(D / 2)[..., None]
    xf, cf = aw.find_map(aw.make_loglik_diag(I, K, fd_tm, fd_oms, True), theta)
    xw, cw = aw.find_map(aw.make_loglik_diag(Y, D, wd_tm[None], wd_oms[None], False), theta)
    sf, sw = np.sqrt(np.diag(cf)), np.sqrt(np.diag(cw))
    out.append(dict(fd=((xf - theta) / sf).tolist(), wdm=((xw - theta) / sw).tolist(),
                    diff=((xw - xf) / sf).tolist(), sig_ratio=(sw / sf).tolist(),
                    zstd_fd=zf.std(0).tolist(), zstd_wdm=zw.std((0, 1)).tolist()))
    print(r, np.round(out[-1]["fd"], 2), np.round(out[-1]["wdm"], 2), np.round(out[-1]["diff"], 3), flush=True)

fd = np.array([o["fd"] for o in out]); wd = np.array([o["wdm"] for o in out])
df = np.array([o["diff"] for o in out]); sr = np.array([o["sig_ratio"] for o in out])
zsf = np.array([o["zstd_fd"] for o in out]); zsw = np.array([o["zstd_wdm"] for o in out])
res = dict(nreal=NREAL, zstd_fd_mean=zsf.mean(0).tolist(), zstd_wdm_mean=zsw.mean(0).tolist(), fd_mean=fd.mean(0).tolist(), fd_std=fd.std(0, ddof=1).tolist(),
           wdm_mean=wd.mean(0).tolist(), wdm_std=wd.std(0, ddof=1).tolist(),
           diff_mean=df.mean(0).tolist(), diff_std=df.std(0, ddof=1).tolist(),
           sigma_ratio_mean=sr.mean(0).tolist(), per_real=out)
print(json.dumps({k: v for k, v in res.items() if k != "per_real"}, indent=1))
os.makedirs("results", exist_ok=True)
with open(OUT, "w") as fh:
    json.dump(res, fh, indent=1)
