"""
Generate the paper's figures as vector PDFs into paper/figures/.

Every figure is built from a CSV already produced by the analysis scripts, so
re-running this after new measurements refreshes the paper without touching the
LaTeX. Figures are sized for a single column of the Elsevier cas-dc layout
(3.4 in wide) unless marked WIDE, which spans both columns (7.0 in).

    fig1_model_landscape   Sec 6.1  nine models vs chance and empirical oracle
    fig2_feature_families  Sec 6.2  five feature families vs drift
    fig3_rank_chain        Sec 6.4  cost ordering through the VND pipeline (WIDE)
    fig4_contraction       Sec 6.5  LKH gain against entering cohort rank
    fig5_kappa_curve       Sec 7.2  calibrated rank C as a function of kappa
    fig6_kappa_observed    Sec 7.3  kappa across 25 benchmark instances
    fig7_per_instance_effect  Sec 5.1  per-instance effect of every model
    fig8_convergence          Sec 5.1  best-so-far, Cox island vs control (WIDE)

Usage
-----
python analysis_la_cetsp/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "analysis_la_cetsp" / "results"
TEST = ROOT / "analysis_test"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = 3.4          # single column width, inches
WIDE = 7.0         # both columns
CHANCE = "#c1121f"
CEIL = "#2a9d8f"
BAR = "#457b9d"
ACCENT = "#e76f51"

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf", flush=True)


# ---------------------------------------------------------------- figure 1
def fig_model_landscape():
    f = RES / "model_ceiling.csv"
    if not f.exists():
        print("  [skip] model_ceiling.csv missing"); return
    d = pd.read_csv(f)
    mean = d[d.instance == "MEAN"].iloc[0]
    models = [c for c in d.columns if c not in ("instance", "n")]
    vals = [(m, float(mean[m])) for m in models]
    vals.sort(key=lambda t: t[1])

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    y = np.arange(len(vals))
    ax.barh(y, [v for _, v in vals], color=BAR, height=0.65, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([m for m, _ in vals])
    ax.axvline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=4, label="chance (0.500)")
    ax.axvline(0.687, color=CEIL, lw=1.2, ls=":", zorder=4, label="empirical oracle (0.687)")
    ax.set_xlim(0.45, 0.72)
    ax.set_xlabel("cross-validated concordance")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    save(fig, "fig1_model_landscape")


def feature_values():
    diversity = pd.read_csv(RES / "diversity_signal.csv").set_index("feature")
    partial = pd.read_csv(RES / "partial_vnd_signal.csv").set_index("stage")
    oracle = pd.read_csv(RES / "oracle_ceiling.csv").set_index("feature_set")
    lineage = pd.read_csv(RES / "lineage_signal.csv").set_index("feature_set")
    return [("Diversity (direct)", diversity.loc["div_min", "C_mean"]),
            ("Greedy (direct)", partial.loc["post_greed_cost", "C_mean"]),
            ("LKH (direct)", partial.loc["post_lkh_cost", "C_mean"]),
            ("Geometry", oracle.loc["PRE_geom", "best"]),
            ("Cohort-relative cost", oracle.loc["PRE_pop", "best"]),
            ("Geometry + cohort", oracle.loc["PRE_all", "best"]),
            ("Lineage", lineage.loc["LINEAGE", "best"]),
            ("All 23 features", lineage.loc["PRE+LINEAGE", "best"])], lineage.loc["DRIFT", "best"]


# ---------------------------------------------------------------- figure 2
def fig_feature_families():
    fams, drift = feature_values()
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    y = np.arange(len(fams))
    colors = [ACCENT if n in ("Lineage", "All 23 features") else BAR for n, _ in fams]
    ax.barh(y, [v for _, v in fams], color=colors, height=0.65, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _ in fams])
    ax.axvline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=4, label="chance")
    ax.axvline(drift, color="k", lw=1.2, ls="-.", zorder=4,
               label=f"lineage clock ({drift:.3f})")
    ax.set_xlim(0.45, 0.66)
    ax.set_xlabel("concordance (protocols differ)")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    save(fig, "fig2_feature_families")


# ---------------------------------------------------------------- figure 3
def fig_rank_chain():
    stages = ["pre-VND", "post-greedy", "post-LKH", "post-VND"]
    d = pd.read_csv(RES / "rank_transmission.csv").set_index("instance")
    data = {inst: row[["greedy_rho", "lkh_rho", "refine_rho"]].tolist()
            for inst, row in d.iterrows()}
    fig, ax = plt.subplots(figsize=(WIDE, 1.9))
    n_row = len(data)
    # stage names as column headers, so nothing collides with the arrows
    for i, s in enumerate(stages):
        ax.text(i, 0.6, s, ha="center", va="center", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="k", lw=0.7))
    for j, (inst, rhos) in enumerate(data.items()):
        y = -0.55 * (j + 1)
        ax.text(-0.55, y, inst, fontsize=7, va="center", ha="right")
        for i, r in enumerate(rhos):
            ax.annotate("", xy=(i + 0.78, y), xytext=(i + 0.22, y),
                        arrowprops=dict(arrowstyle="-|>", lw=0.6 + 3.0 * r,
                                        color=plt.cm.viridis(1 - r), alpha=0.9))
            ax.text(i + 0.5, y + 0.14, f"{r:+.2f}", ha="center", fontsize=7)
        # tick marks under each stage column to anchor the rows visually
        for i in range(len(stages)):
            ax.plot([i], [y], marker="|", ms=4, color="0.6", zorder=1)
    ax.set_xlim(-1.5, len(stages) - 0.4)
    ax.set_ylim(-0.55 * (n_row + 0.6), 0.95)
    ax.axis("off")
    ax.set_title("Spearman correlation of the offspring cost ordering, stage to stage",
                 pad=1, fontsize=8)
    save(fig, "fig3_rank_chain")


# ---------------------------------------------------------------- figure 4
def fig_contraction():
    # Points and annotations share the exact run selection and estimator.
    points = pd.read_csv(RES / "contraction_points.csv")
    kappas = pd.read_csv(RES / "contraction_kappa.csv").set_index("instance")
    panels = [(inst, g.cohort_rank.to_numpy(), g.relative_gain.to_numpy(),
               kappas.loc[inst, "kappa"]) for inst, g in points.groupby("instance")]
    fig, axes = plt.subplots(1, len(panels), figsize=(WIDE, 2.1), sharey=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (inst, x, y, rho) in zip(axes, panels):
        ax.scatter(x, y, s=3, alpha=0.25, color=BAR, edgecolors="none")
        # binned median, to make the trend legible through the cloud
        bins = np.linspace(0, 1, 11)
        idx = np.minimum(np.digitize(x, bins) - 1, 9)
        bx, by = [], []
        for k in range(10):
            m = idx == k
            if m.sum() > 5:
                bx.append(0.5 * (bins[k] + bins[k + 1])); by.append(np.median(y[m]))
        ax.plot(bx, by, color=ACCENT, lw=1.6, zorder=4)
        ax.set_title(f"{inst}\n$\\kappa_{{\\mathrm{{LKH}}}}={rho:+.2f}$", fontsize=7.5)
        ax.set_xlabel("cohort cost rank entering LKH")
        ax.grid(lw=0.35, alpha=0.35)
    axes[0].set_ylabel("relative gain from LKH")
    save(fig, "fig4_contraction")


# ---------------------------------------------------------------- figure 5
def fig_kappa_curve():
    f = RES / "kappa_calibration.csv"
    if not f.exists():
        print("  [skip] kappa_calibration.csv missing"); return
    d = pd.read_csv(f)
    obs = RES / "kappa_observed.csv"
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot(d.kappa_mean, d.C_mean, "-o", color=BAR, ms=3.5, lw=1.4, zorder=3)
    ax.fill_between(d.kappa_mean, d.C_mean - d.C_std, d.C_mean + d.C_std,
                    color=BAR, alpha=0.18, zorder=2)
    ax.axhline(0.5, color=CHANCE, lw=1.0, ls="--", zorder=1, label="chance")
    if obs.exists():
        o = pd.read_csv(obs)
        k, c = o.kappa_observed.mean(), o.C_observed.mean()
        ax.scatter([k], [c], s=45, color=ACCENT, zorder=5, marker="D",
                   label=f"measured system\n($\\kappa$={k:.2f}, C={c:.3f})")
    ax.set_xlabel(r"VND rank--gain coefficient $\kappa_{\mathrm{VND}}$")
    ax.set_ylabel("cost-rank concordance")
    ax.set_ylim(0.48, 0.76)
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax.grid(lw=0.4, alpha=0.4)
    save(fig, "fig5_kappa_curve")


# ---------------------------------------------------------------- figure 6
def fig_kappa_observed():
    f = RES / "kappa_observed.csv"
    if not f.exists():
        print("  [skip] kappa_observed.csv missing"); return
    d = pd.read_csv(f).sort_values("kappa_observed")
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    y = np.arange(len(d))
    ax.hlines(y, 0, d.kappa_observed, color=BAR, lw=1.1, zorder=2)
    ax.scatter(d.kappa_observed, y, s=12, color=BAR, zorder=3)
    ax.axvline(d.kappa_observed.mean(), color=ACCENT, lw=1.2, ls="--", zorder=4,
               label=f"mean {d.kappa_observed.mean():.3f}")
    ax.set_yticks(y)
    ax.set_yticklabels(d.instance, fontsize=5.5)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(r"VND rank--gain coefficient $\kappa_{\mathrm{VND}}$")
    ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax.grid(axis="x", lw=0.4, alpha=0.4)
    save(fig, "fig6_kappa_observed")


# ---------------------------------------------------------------- figure 7
def fig_per_instance_effect():
    """Per-instance mean-cost ratios, one treatment at a time."""
    f = TEST / "parsed" / "island_summaries.csv"
    if not f.exists():
        print("  [skip] island_summaries.csv missing"); return
    d = pd.read_csv(f)
    per = d.groupby(["instance", "model"]).best.mean().unstack()
    if "BASELINE" not in per.columns:
        print("  [skip] no BASELINE column"); return
    base = per["BASELINE"]
    models = [m for m in per.columns if m != "BASELINE"]
    diff = per[models].sub(base, axis=0).div(base, axis=0) * 100.0
    order = diff.median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(COL, 2.9))
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = diff[m].dropna().values
        ax.scatter(v, np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)),
                   s=6, alpha=0.45, color=BAR, edgecolors="none", zorder=3)
        ax.scatter([np.median(v)], [i], marker="|", s=180, color=ACCENT,
                   zorder=5, lw=1.6)
    ax.axvline(0, color="k", lw=1.0, zorder=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    lim = np.nanpercentile(np.abs(diff.values), 97)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("cost relative to control (%), per instance")
    ax.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    ax.set_title("negative = better than control", fontsize=7.5, pad=3)
    save(fig, "fig7_per_instance_effect")


def fig_convergence():
    f = TEST / "parsed" / "convergence.csv"
    if not f.exists():
        print("  [skip] convergence.csv missing"); return
    d = pd.read_csv(f)
    # Parallel islands share stdout without a lock, so a minority of [LOG] lines
    # interleave and the parser recovers impossible iteration numbers. The budget
    # is 5000, so anything outside [1, 5000] is corruption and is dropped.
    d = d[(d.iter >= 1) & (d.iter <= 5000) & (d.best > 0)]

    counts = d.groupby("instance").iter.max().sort_values(ascending=False)
    picks = [i for i in ["car_door_50", "d493_or30", "team4_400"] if i in counts.index]
    picks += [i for i in counts.index if i not in picks][:3 - len(picks)]
    picks = picks[:3]

    XMAX = 2200          # runs stop on stagnation well before the 5000 budget
    fig, axes = plt.subplots(1, len(picks), figsize=(WIDE, 2.2))
    if len(picks) == 1:
        axes = [axes]
    for ax, inst in zip(axes, picks):
        sub = d[(d.instance == inst) & (d.iter <= XMAX)]
        base = sub[sub.island == 9].groupby("iter").best.mean()
        # Show one named treatment against its no-filter control.
        mlm = sub[sub.island == 0].groupby("iter").best.mean()
        if base.empty or mlm.empty:
            continue
        # express as excess over the best value either arm reaches, so the
        # comparison is visible instead of being buried by the initial descent
        floor = min(base.min(), mlm.min())
        ax.plot(base.index, 100 * (base.values - floor) / floor,
                color="k", lw=1.2, label="control (no ML)")
        ax.plot(mlm.index, 100 * (mlm.values - floor) / floor,
                color=ACCENT, lw=1.1, alpha=0.9, label="Cox PH island")
        ax.axvline(1000, color=CEIL, lw=1.0, ls=":", label="filter activates")
        ax.set_yscale("log")
        ax.set_title(inst, fontsize=7.5)
        ax.set_xlabel("iteration")
        ax.set_xlim(0, XMAX)
        ax.grid(lw=0.35, alpha=0.35, which="both")
    axes[0].set_ylabel("excess over best (%)")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    save(fig, "fig8_convergence")


# --------------------------------------------------- merged: 1+2 and 5+6
# Two single-column figures placed as separate floats produced float pages:
# LaTeX could not fit text alongside them and gave each pair a page of its own.
# Merging each natural pair into ONE full-width float halves the float count and
# leaves the rest of the page for text.

def fig_landscape_combined():
    """Model landscape and feature families, side by side, full width."""
    fm = RES / "model_ceiling.csv"
    if not fm.exists():
        print("  [skip] model_ceiling.csv missing"); return
    d = pd.read_csv(fm)
    mean = d[d.instance == "MEAN"].iloc[0]
    models = sorted(((m, float(mean[m])) for m in d.columns if m not in ("instance", "n")),
                    key=lambda t: t[1])

    fams, drift = feature_values()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDE, 2.6))

    y = np.arange(len(models))
    ax1.barh(y, [v for _, v in models], color=BAR, height=0.68, zorder=3)
    ax1.set_yticks(y); ax1.set_yticklabels([m for m, _ in models])
    ax1.axvline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=4, label="chance")
    ax1.axvline(0.687, color=CEIL, lw=1.2, ls=":", zorder=4, label="empirical oracle")
    ax1.set_xlim(0.45, 0.72)
    ax1.set_xlabel("cross-validated concordance")
    ax1.set_title("(a) by model", fontsize=8)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax1.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)

    y = np.arange(len(fams))
    colors = [ACCENT if n in ("Lineage", "All 23 features") else BAR for n, _ in fams]
    ax2.barh(y, [v for _, v in fams], color=colors, height=0.68, zorder=3)
    ax2.set_yticks(y); ax2.set_yticklabels([n for n, _ in fams])
    ax2.axvline(0.5, color=CHANCE, lw=1.2, ls="--", zorder=4, label="chance")
    ax2.axvline(drift, color="k", lw=1.2, ls="-.", zorder=4,
                label=f"lineage clock ({drift:.3f})")
    ax2.set_xlim(0.45, 0.66)
    ax2.set_xlabel("concordance (protocols differ)")
    ax2.set_title("(b) by feature family", fontsize=8)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax2.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)

    fig.tight_layout(w_pad=2.0)
    save(fig, "figA_landscape")


def fig_kappa_combined():
    """Calibration curve and the observed spread, side by side, full width."""
    fc, fo = RES / "kappa_calibration.csv", RES / "kappa_observed.csv"
    if not fc.exists():
        print("  [skip] kappa_calibration.csv missing"); return
    d = pd.read_csv(fc)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDE, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.15]})

    ax1.plot(d.kappa_mean, d.C_mean, "-o", color=BAR, ms=3.5, lw=1.4, zorder=3)
    ax1.fill_between(d.kappa_mean, d.C_mean - d.C_std, d.C_mean + d.C_std,
                     color=BAR, alpha=0.18, zorder=2)
    ax1.axhline(0.5, color=CHANCE, lw=1.0, ls="--", zorder=1, label="chance")
    if fo.exists():
        o = pd.read_csv(fo)
        k, c = o.kappa_observed.mean(), o.C_observed.mean()
        ax1.scatter([k], [c], s=45, color=ACCENT, zorder=5, marker="D",
                    label=f"measured ($\\kappa$={k:.2f}, C={c:.3f})")
    ax1.set_xlabel(r"VND rank--gain coefficient $\kappa_{\mathrm{VND}}$")
    ax1.set_ylabel("cost-rank concordance")
    ax1.set_ylim(0.48, 0.76)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
    ax1.grid(lw=0.4, alpha=0.4)
    ax1.set_title("(a) fixed-membership sensitivity", fontsize=8)

    if fo.exists():
        o = pd.read_csv(fo).sort_values("kappa_observed")
        y = np.arange(len(o))
        ax2.hlines(y, 0, o.kappa_observed, color=BAR, lw=1.0, zorder=2)
        ax2.scatter(o.kappa_observed, y, s=10, color=BAR, zorder=3)
        ax2.axvline(o.kappa_observed.mean(), color=ACCENT, lw=1.2, ls="--",
                    zorder=4, label=f"mean {o.kappa_observed.mean():.3f}")
        ax2.set_yticks(y); ax2.set_yticklabels(o.instance, fontsize=7)
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel(r"VND rank--gain coefficient $\kappa_{\mathrm{VND}}$")
        ax2.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
        ax2.grid(axis="x", lw=0.4, alpha=0.4)
        ax2.set_title("(b) measured on 25 instances", fontsize=8)

    fig.tight_layout(w_pad=1.6)
    save(fig, "figB_kappa")


def fig_results_combined():
    """Section 5 in one full-width float: the null at instance resolution, and a
    representative trajectory with the filter activation point marked.

    Replaces the separate per-instance and convergence figures, which as two
    floats in a short section each ended up alone on a page.
    """
    fs = TEST / "parsed" / "island_summaries.csv"
    fc = TEST / "parsed" / "convergence.csv"
    if not fs.exists():
        print("  [skip] island_summaries.csv missing"); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDE, 2.7),
                                   gridspec_kw={"width_ratios": [1, 1.05]})

    # ---- (a) per-instance effect ------------------------------------------
    d = pd.read_csv(fs)
    per = d.groupby(["instance", "model"]).best.mean().unstack()
    base = per["BASELINE"]
    models = [m for m in per.columns if m != "BASELINE"]
    diff = per[models].sub(base, axis=0).div(base, axis=0) * 100.0
    order = diff.median().sort_values().index.tolist()
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        v = diff[m].dropna().values
        ax1.scatter(v, np.full(len(v), i) + rng.uniform(-0.16, 0.16, len(v)),
                    s=5, alpha=0.45, color=BAR, edgecolors="none", zorder=3)
        ax1.scatter([np.median(v)], [i], marker="|", s=150, color=ACCENT,
                    zorder=5, lw=1.5)
    ax1.axvline(0, color="k", lw=1.0, zorder=4)
    ax1.set_yticks(range(len(order))); ax1.set_yticklabels(order, fontsize=6.5)
    lim = np.nanpercentile(np.abs(diff.values), 97)
    ax1.set_xlim(-lim, lim)
    ax1.set_xlabel("cost relative to control (%), per instance")
    ax1.grid(axis="x", lw=0.4, alpha=0.4, zorder=0)
    ax1.set_title("(a) per-instance effect (negative = better)", fontsize=8)

    # ---- (b) one representative trajectory --------------------------------
    if fc.exists():
        c = pd.read_csv(fc)
        c = c[(c.iter >= 1) & (c.iter <= 5000) & (c.best > 0)]
        # Pick an instance whose runs continue well past the activation point,
        # otherwise the filter never acts within the plotted window and the
        # panel shows nothing. bubbles8: control stops ~3344, Cox ~3659.
        for cand in ("bubbles8", "bubbles9", "team3_300"):
            if cand in set(c.instance):
                pick = cand
                break
        else:
            pick = c.instance.iloc[0]
        sub = c[(c.instance == pick) & (c.iter <= 3600)]
        bs = sub[sub.island == 9].groupby("iter").best.mean()
        cx = sub[sub.island == 0].groupby("iter").best.mean()
        floor = min(bs.min(), cx.min())
        # clamp: excess is exactly zero once an arm reaches the floor, and log(0)
        # sends the curve off-scale as a spurious vertical spike
        eps = 1e-3
        ax2.plot(bs.index, np.maximum(100 * (bs.values - floor) / floor, eps),
                 color="k", lw=1.2, label="control (no ML)")
        ax2.plot(cx.index, np.maximum(100 * (cx.values - floor) / floor, eps),
                 color=ACCENT, lw=1.1, alpha=0.9, label="Cox PH island")
        ax2.axvline(1000, color=CEIL, lw=1.1, ls=":", label="filter activates")
        ax2.set_yscale("log"); ax2.set_ylim(bottom=eps)
        ax2.set_xlabel("iteration"); ax2.set_ylabel("excess over best (%)")
        ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
        ax2.grid(lw=0.35, alpha=0.35, which="both")
        ax2.set_title(f"(b) trajectory, {pick}", fontsize=8)

    fig.tight_layout(w_pad=1.8)
    save(fig, "figC_results")


if __name__ == "__main__":
    print(f"writing figures to {OUT}")
    for fn in (fig_model_landscape, fig_feature_families, fig_rank_chain,
               fig_contraction, fig_kappa_curve, fig_kappa_observed,
               fig_per_instance_effect, fig_convergence,
               fig_landscape_combined, fig_kappa_combined, fig_results_combined):
        fn()
    print("done")
