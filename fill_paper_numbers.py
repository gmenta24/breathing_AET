"""Fill the result-dependent placeholders of paper/main_new.tex from results/summary.json."""

import json
import sys

import numpy as np

src = sys.argv[1] if len(sys.argv) > 1 else "../paper/main_new.tex"
d = json.load(open("results/summary.json"))
P, T = d["posteriors"], d["timing"]

rows = [
    ("Analysis 1", "FD, diagonal $AET$", "A1: FD, diagonal AET"),
    ("", r"\WDM, diagonal $AET$", "A1: WDM, diagonal AET"),
    ("Analysis 2", r"\WDM, time-local $3\times3$", "A2: WDM, local 3x3"),
    ("", r"FD, yearly average $3\times3$", "A2: FD, time-averaged 3x3"),
    ("", r"FD, arms frozen at mid-year", "A2: FD, frozen arms 3x3"),
    ("Analysis 3", r"FD, equal-arm diagonal $AET$", "A3: FD, equal-arm diag AET"),
    ("", r"\WDM, equal-arm diagonal $AET$", "A3: WDM, equal-arm diag AET"),
    ("", r"FD, equal-arm diagonal $AE$", "A3: FD, equal-arm diag AE only"),
]


def fmt_bias(b):
    return f"{b:+.1f}" if abs(b) < 10 else f"{b:+.0f}"


lines = []
for i, (grp, lab, key) in enumerate(rows):
    v = P[key]
    if grp and i:
        lines.append(r"\midrule")
    lines.append(f"{grp} & {lab} & ${v['mean'][0]:.4f}\\pm{v['std'][0]:.4f}$ & "
                 f"${v['mean'][1]:.4f}\\pm{v['std'][1]:.4f}$ & "
                 f"$({fmt_bias(v['bias_sigma'][0])},\\,{fmt_bias(v['bias_sigma'][1])})$ & "
                 f"{v['t_eval_ms']:.2f}\\\\")

dm = {int(k): v / 1e3 for k, v in T["fd_dense_measured_ms"].items()}
Ks = np.array(sorted(dm))
ts = np.array([dm[k] for k in Ks])
slope = np.polyfit(np.log(Ks[-3:]), np.log(ts[-3:]), 1)[0]
Kf = T["K_full"]
t3 = ts[-1] * (Kf / Ks[-1]) ** 3
tfit = ts[-1] * (Kf / Ks[-1]) ** slope
dense = (f"{ts[Ks == 2048][0]:.1f}~s for $K=2048$ bins (a $6144\\times6144$ complex matrix) and "
         f"{ts[-1]:.0f}~s for $K=4096$.  Between these sizes the time grows as $K^{{{slope:.1f}}}$, still below the "
         f"asymptotic $K^3$ of the factorisation; extrapolating to the $K={Kf:,}$ bins of our band gives between "
         f"{tfit / 86400:.0f}~days ($K^{{{slope:.1f}}}$) and {t3 / 3.156e7:.1f}~years ($K^3$) per evaluation, for a matrix of "
         f"{T['dense_memory_TB']:.0f}~TB.").replace(",", "\\,")

s = open(src).read()
s = s.replace("%%TABLE-ROWS%%", "\n".join(lines))
s = s.replace("%%DENSE-TEXT%%", dense)
open(src, "w").write(s)
print(dense)
print("\n".join(lines))
print("wdm cells %.2f ms, fd cells %.2f ms, wdm fine %.0f ms, fd fine %.0f ms" %
      (T["wdm_cells"], T["fd_cells"], T["wdm_fine"], T["fd_fine_blockdiag"]))
