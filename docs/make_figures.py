"""Regenerate the figures in docs/images from the library itself.

    pip install -e . matplotlib
    python docs/make_figures.py
"""

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statlab
from statlab import dist

OUT = Path(__file__).resolve().parent / "images"
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False})


def distributions() -> None:
    xs = [i / 50 for i in range(-250, 251)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(xs, [dist.norm_pdf(x) for x in xs], "k", lw=2, label="normal")
    for df in (1, 3, 10):
        a1.plot(xs, [dist.t_pdf(x, df) for x in xs], label=f"Student t, df = {df}")
    q = dist.t_ppf(0.975, 10)
    a1.fill_between([x for x in xs if x >= q], [dist.t_pdf(x, 10) for x in xs if x >= q], alpha=0.3)
    a1.annotate(f"2.5 % tail, t = {q:.3f}", (q + 0.3, 0.02), xytext=(2.4, 0.15), arrowprops={"arrowstyle": "->"})
    a1.set(xlabel="x", ylabel="density", title="t densities and quantile from the incomplete beta function")
    a1.legend()
    ks = range(0, 21)
    a2.bar([k - 0.2 for k in ks], [dist.binom_pmf(k, 20, 0.3) for k in ks], width=0.4, label="binomial n = 20, p = 0.3")
    a2.bar([k + 0.2 for k in ks], [dist.poisson_pmf(k, 6) for k in ks], width=0.4, label="Poisson, lambda = 6")
    a2.set(xlabel="k", ylabel="P(X = k)", title="Discrete distributions with the same mean")
    a2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "distributions.png")


def regression() -> None:
    rng = random.Random(4)
    x = [rng.uniform(0, 10) for _ in range(40)]
    y = [2.0 + 0.8 * a + rng.gauss(0, 1.2) for a in x]
    r = statlab.linear_regression(x, y)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x, y, s=18, label="synthetic sample, n = 40")
    ax.plot([0, 10], [r["intercept"], r["intercept"] + 10 * r["slope"]], "r", lw=2,
            label=f"y = {r['intercept']:.2f} + {r['slope']:.3f} x   R2 = {r['r2']:.3f}")
    for s in (r["slope_ci_low"], r["slope_ci_high"]):
        ax.plot([0, 10], [r["intercept"], r["intercept"] + 10 * s], "r:", lw=1)
    p_text = "p < 1e-6" if r["p_slope"] < 1e-6 else f"p = {r['p_slope']:.2g}"
    ax.set(xlabel="x", ylabel="y",
           title=f"Least squares, 95 % CI of the slope [{r['slope_ci_low']:.3f}, {r['slope_ci_high']:.3f}], {p_text}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "regression.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    distributions()
    regression()
