import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os
HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "Helvetica Neue", "font.size": 15,
    "axes.edgecolor": "#B9BEC6", "axes.linewidth": 0.9,
    "text.color": "#1D2330", "axes.labelcolor": "#1D2330",
    "xtick.color": "#4A5160", "ytick.color": "#4A5160",
    "figure.facecolor": "white", "savefig.facecolor": "white",
})
TEAL, GREY, CORAL, INK = "#12867E", "#A7ADB8", "#C0453B", "#1D2330"

fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.3),
                               gridspec_kw={"width_ratios": [1.65, 1]})

# ── left: per-instance EPR at 100 / 200 / 360 qubits ─────────────────────────
labels = ["QFT\n100q", "QFT\n200q", "QFT\n360q",
          "QPE\n100q", "QPE\n200q", "QPE\n360q"]
ts = [248, 1232, None, 586, None, None]
ds = [201,  766, 1890, 246,  896, 1938]

x, w = np.arange(len(labels)), 0.38
for i, v in enumerate(ts):
    if v is None:
        axL.bar(i - w/2, 2100, w, color="none", edgecolor=CORAL,
                linestyle=(0, (2.5, 2)), linewidth=1.3, zorder=3)
        axL.text(i - w/2, 1020, "✗", ha="center", va="center",
                 fontsize=20, color=CORAL, fontweight="bold",
                 fontname="Arial Unicode MS")
    else:
        axL.bar(i - w/2, v, w, color=GREY, zorder=3)
for i, v in enumerate(ds):
    axL.bar(i + w/2, v, w, color=TEAL, zorder=3)

axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=12)
axL.set_ylim(0, 2450)
axL.set_ylabel("EPR pairs", fontsize=13)
axL.set_title("Scaling up:  6, 12 and 20 cores", fontsize=14.5, pad=26)
axL.grid(axis="y", color="#E5E8ED"); axL.set_axisbelow(True)
for s in ("top", "right"): axL.spines[s].set_visible(False)

h1 = plt.Rectangle((0, 0), 1, 1, color=GREY)
h2 = plt.Rectangle((0, 0), 1, 1, color=TEAL)
h3 = plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=CORAL,
                   linestyle=(0, (2.5, 2)), linewidth=1.3)
axL.legend([h1, h2, h3], ["TeleSABRE", "dSABRE", "TeleSABRE: no result"],
           frameon=False, fontsize=11.5, ncol=3, loc="upper center",
           bbox_to_anchor=(0.5, 1.13), handlelength=1.5,
           columnspacing=1.2, handletextpad=0.5)


# ── right: completion ────────────────────────────────────────────────────────
axR.bar(["dSABRE", "TeleSABRE"], [45, 45], color="#EDF0F4", width=0.52, zorder=2)
axR.bar(["dSABRE", "TeleSABRE"], [45, 40], color=[TEAL, GREY], width=0.52, zorder=3)
for i, v in enumerate([45, 40]):
    axR.text(i, v - 4.5, f"{v} / 45", ha="center", fontsize=17,
             fontweight="bold", color="white")
axR.set_ylim(0, 50)
axR.set_ylabel("instances routed", fontsize=13)
axR.set_title("Did it finish at all?", fontsize=14.5, pad=26)
axR.grid(axis="y", color="#E5E8ED"); axR.set_axisbelow(True)
for s in ("top", "right"): axR.spines[s].set_visible(False)
axR.tick_params(axis="x", labelsize=14)

fig.tight_layout()
fig.savefig(f"{HERE}/chart_scale.png", dpi=220)
print("ok")
