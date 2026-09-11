import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

HERE = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "Helvetica Neue",
    "font.size": 15,
    "axes.edgecolor": "#B9BEC6",
    "axes.linewidth": 0.9,
    "text.color": "#1D2330",
    "axes.labelcolor": "#1D2330",
    "xtick.color": "#4A5160",
    "ytick.color": "#4A5160",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

TEAL   = "#12867E"
GREY   = "#A7ADB8"
CORAL  = "#C0453B"
INK    = "#1D2330"

# ── 1. headline: % EPR reduction across the five evaluated settings ──────────
labels = ["25 qubits\nB-grid (4 cores)",
          "36 qubits\nB-grid (4 cores)",
          "64 qubits\nH-grid (6 cores)",
          "64 qubits\nheavy-hex ring",
          "64 qubits\nheavy-hex star"]
red = [50.8, 62.0, 49.1, 52.0, 41.5]

fig, ax = plt.subplots(figsize=(11.2, 4.5))
y = np.arange(len(labels))[::-1]
bars = ax.barh(y, red, height=0.60, color=TEAL, zorder=3)
for yi, v in zip(y, red):
    ax.text(v + 1.4, yi, f"−{v:.1f}%", va="center", ha="left",
            fontsize=17, fontweight="bold", color=TEAL)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=13.5)
ax.set_xlim(0, 76)
ax.set_xlabel("Reduction in EPR pairs used, vs TeleSABRE  (geometric mean)",
              fontsize=13.5, labelpad=10)
ax.xaxis.set_major_formatter(lambda v, p: f"{v:.0f}%")
ax.grid(axis="x", color="#E5E8ED", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(f"{HERE}/chart_headline.png", dpi=220)
plt.close(fig)

# ── 2. per-circuit, 64-qubit suite ───────────────────────────────────────────
circ = ["GHZ", "Graph-\nstate", "AE", "QFT", "QPE-\nexact", "QAOA", "QNN",
        "Multi-\nplier", "Random"]
ts   = [16, 62, 353, 312, 289, 956, 889, 1601, np.nan]
ds   = [6, 15, 165, 180, 187, 555, 539, 1285, 732]

fig, ax = plt.subplots(figsize=(11.6, 4.6))
x = np.arange(len(circ)); w = 0.38
ax.bar(x - w/2, ts, w, label="TeleSABRE", color=GREY, zorder=3)
ax.bar(x + w/2, ds, w, label="dSABRE", color=TEAL, zorder=3)
ax.set_yscale("log")
ax.set_ylim(3, 4200)
ax.set_ylabel("EPR pairs consumed  (log scale)", fontsize=13.5, labelpad=8)
ax.set_xticks(x); ax.set_xticklabels(circ, fontsize=12.5)
ax.grid(axis="y", color="#E5E8ED", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
h1 = plt.Rectangle((0, 0), 1, 1, color=GREY)
h2 = plt.Rectangle((0, 0), 1, 1, color=TEAL)
h3 = plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=CORAL,
                   linestyle=(0, (2.5, 2)), linewidth=1.3)
ax.legend([h1, h2, h3], ["TeleSABRE", "dSABRE", "TeleSABRE: no result"],
          frameon=False, fontsize=13, ncol=3, loc="upper left",
          handlelength=1.5, columnspacing=1.4, handletextpad=0.55)
# TeleSABRE fails on Random: draw an empty slot rather than a missing bar
ax.bar(len(circ)-1 - w/2, 900, w, color="none", edgecolor=CORAL,
       linestyle=(0, (2.5, 2)), linewidth=1.3, zorder=4)
ax.scatter([len(circ)-1 - w/2], [70], marker="x", s=150, color=CORAL,
           linewidths=2.6, zorder=5)
fig.tight_layout()
fig.savefig(f"{HERE}/chart_percircuit.png", dpi=220)
plt.close(fig)

# ── 3. ablation: what the capacity mechanisms are worth ──────────────────────
abl = ["Full dSABRE",
       "Drop the lookahead ordering",
       "Drop the capacity penalty",
       "Drop the hard capacity floor",
       "Drop both capacity mechanisms"]
delta = [0.0, 12.4, 9.4, 0.2, 87.8]
aborts = ["", "", "", "", "2 circuits\nnever finish"]

fig, ax = plt.subplots(figsize=(11.2, 4.4))
y = np.arange(len(abl))[::-1]
cols = [TEAL] + [GREY]*3 + [CORAL]
ax.barh(y, delta, height=0.58, color=cols, zorder=3)
for yi, v, lab in zip(y, delta, aborts):
    ax.text(v + 1.6, yi, f"+{v:.1f}%" if v else "baseline",
            va="center", ha="left", fontsize=15.5, fontweight="bold",
            color=CORAL if v > 50 else INK)
    if lab:
        ax.text(v + 15.5, yi, lab, va="center", ha="left",
                fontsize=12, color=CORAL)
ax.set_yticks(y); ax.set_yticklabels(abl, fontsize=13.5)
ax.set_xlim(0, 118)
ax.set_xlabel("Extra EPR pairs needed when the component is removed  (64-qubit suite)",
              fontsize=13.5, labelpad=10)
ax.xaxis.set_major_formatter(lambda v, p: f"+{v:.0f}%")
ax.grid(axis="x", color="#E5E8ED", zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(f"{HERE}/chart_ablation.png", dpi=220)
plt.close(fig)

# ── 4. scale/completion lives in chart_scale.py — do not duplicate it here ───
print("charts written")
