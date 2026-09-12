import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                 "font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 300, "mathtext.fontset": "stix"})
OUT = "paper/figs_png"; os.makedirs(OUT, exist_ok=True)
W = 3.4
BLUE = "#4C72B0"; RED = "#C44E52"; GRAY = "#888"; LGRAY = "#aaa"
STATES = ["NC", "PA", "MD"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUC = {
    "NC": {"learned": (0.868, 0.849), "hand": (0.826, 0.787)},
    "PA": {"learned": (0.900, 0.953), "hand": (0.830, 0.883)},
    "MD": {"learned": (0.789, 0.846), "hand": (0.783, 0.841)},
}
fig, ax = plt.subplots(figsize=(W, 2.4))
ypos = {"NC": 3, "PA": 2, "MD": 1}
for s in STATES:
    y = ypos[s]; a = AUC[s]
    ax.plot(a["learned"][0], y + 0.12, marker="o", color=BLUE, ms=5, ls="")
    ax.plot(a["learned"][1], y + 0.04, marker="s", color=BLUE, ms=5, ls="")
    ax.plot(a["hand"][0], y - 0.04, marker="o", mfc="white", mec=RED, ms=5, ls="")
    ax.plot(a["hand"][1], y - 0.12, marker="s", mfc="white", mec=RED, ms=5, ls="")
ax.set_yticks([1, 2, 3]); ax.set_yticklabels(["MD", "PA", "NC"])
ax.set_xlim(0.74, 0.98); ax.set_ylim(0.55, 3.5)
ax.set_xlabel("Node-localization AUC")
ax.grid(axis="x", lw=0.3, alpha=0.5)
leg = [Line2D([0], [0], marker="o", color=BLUE, ls="", ms=5, label="learned, decorr."),
       Line2D([0], [0], marker="s", color=BLUE, ls="", ms=5, label="learned, correlated"),
       Line2D([0], [0], marker="o", mfc="white", mec=RED, color=RED, ls="", ms=5, label="hand geometry, decorr."),
       Line2D([0], [0], marker="s", mfc="white", mec=RED, color=RED, ls="", ms=5, label="hand geometry, correlated")]
ax.legend(handles=leg, fontsize=6, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          handletextpad=0.3, columnspacing=1.4)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_localization.png", bbox_inches="tight"); plt.close()

enacted = {"NC": 749, "PA": 2367, "MD": 777}
titles = {"NC": "NC: enacted in the bulk",
          "PA": r"PA: enacted past the tail ($+5.8\sigma$)",
          "MD": r"MD: enacted past the tail ($+13.7\sigma$)"}
fig, axes = plt.subplots(3, 1, figsize=(W, 3.3))
for ax, s in zip(axes, STATES):
    ce = pd.read_csv(f"{ROOT}/paper/ensemble_{s}.csv")["cut_edges"].values
    ax.hist(ce, bins=22, color=BLUE, edgecolor="white", linewidth=0.3)
    ax.axvline(enacted[s], color=RED, lw=1.6)
    lo = min(ce.min(), enacted[s]); hi = max(ce.max(), enacted[s]); pad = 0.05 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_title(titles[s], fontsize=7, loc="left", pad=2)
    ax.set_yticks([]); ax.grid(axis="x", lw=0.3, alpha=0.4); ax.tick_params(labelsize=6.5)
axes[-1].set_xlabel("cut edges (enacted plan in red)")
fig.tight_layout(h_pad=0.9); fig.savefig(f"{OUT}/fig3_cutedges.png", bbox_inches="tight"); plt.close()

import networkx as nx
G = nx.grid_2d_graph(6, 3)
pos = {n: (n[0], n[1]) for n in G.nodes()}
def district(x, y):
    if x <= 1 or (x, y) == (2, 1): return 0
    if x <= 3: return 1
    return 2
dist = {n: district(*n) for n in G.nodes()}
DCOL = {0: "#4C72B0", 1: "#55A868", 2: "#8172B3"}
fig, ax = plt.subplots(figsize=(W, 1.85))
for u, v in G.edges():
    cut = dist[u] != dist[v]
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=(RED if cut else "#cccccc"), lw=(1.9 if cut else 0.8), zorder=1)
for n in G.nodes():
    ax.plot(pos[n][0], pos[n][1], "o", ms=10, color=DCOL[dist[n]],
            markeredgecolor="white", markeredgewidth=0.7, zorder=2)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_aspect("equal"); ax.margins(0.08)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_toygraph.png", bbox_inches="tight"); plt.close()
print("wrote fig2_localization, fig3_cutedges, fig_toygraph in procurement style")
