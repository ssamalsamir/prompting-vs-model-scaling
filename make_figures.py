"""Generate paper figures from the aggregated results."""
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HOME = Path.home()
OUT = HOME / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

agg = pd.read_csv(HOME / "results" / "aggregate_accuracy.csv")

families = ["Qwen2.5-Instruct", "Llama-3"]
tasks = ["sst2", "gsm8k", "mmlu"]
task_titles = {"sst2": "SST-2 (sentiment)", "gsm8k": "GSM8K (math reasoning)",
               "mmlu": "MMLU (knowledge)"}
strategies = ["zero_shot", "few_shot", "cot", "structured"]
slabel = {"zero_shot": "zero-shot", "few_shot": "few-shot", "cot": "CoT",
          "structured": "structured"}
colors = {"zero_shot": "#7f7f7f", "few_shot": "#1f77b4", "cot": "#d62728",
          "structured": "#2ca02c"}
markers = {"zero_shot": "o", "few_shot": "s", "cot": "^", "structured": "D"}


def fmt_log(ax, xs):
    ax.set_xscale("log")
    ax.set_xticks(sorted(xs))
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mticker.NullFormatter())


# ── Figure 1: scaling curves grid (families x tasks) ──
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for i, fam in enumerate(families):
    for j, task in enumerate(tasks):
        ax = axes[i][j]
        sub = agg[(agg.family == fam) & (agg.benchmark == task)]
        for s in strategies:
            ss = sub[sub.strategy == s].sort_values("params_b")
            if len(ss):
                ax.plot(ss.params_b, ss.acc_mean, marker=markers[s], color=colors[s],
                        label=slabel[s], lw=2, ms=6)
        fmt_log(ax, sub.params_b.unique())
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.set_title(task_titles[task], fontsize=11)
        if j == 0:
            ax.set_ylabel(f"{fam}\naccuracy (%)", fontsize=10)
        if i == 1:
            ax.set_xlabel("parameters (B, log scale)")
h, l = axes[0][0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
fig.suptitle("Figure 1.  Accuracy vs. model scale, by prompting strategy",
             y=1.05, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig1_scaling.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: CoT − zero-shot delta vs size, per task, both families ──
fam_color = {"Qwen2.5-Instruct": "#6a3d9a", "Llama-3": "#ff7f00"}
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for j, task in enumerate(tasks):
    ax = axes[j]
    for fam in families:
        sub = agg[(agg.family == fam) & (agg.benchmark == task)]
        piv = sub.pivot_table(index="params_b", columns="strategy", values="acc_mean")
        if "cot" in piv and "zero_shot" in piv:
            delta = (piv["cot"] - piv["zero_shot"]).sort_index()
            ax.plot(delta.index, delta.values, marker="o", color=fam_color[fam],
                    label=fam, lw=2.5, ms=7)
    ax.axhline(0, color="k", lw=0.9, ls="--")
    fmt_log(ax, agg[agg.benchmark == task].params_b.unique())
    ax.tick_params(axis="x", labelsize=8)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30); lbl.set_ha("right")
    ax.set_title(task_titles[task])
    ax.grid(alpha=0.3)
    ax.set_xlabel("parameters (B, log scale)")
axes[0].set_ylabel("CoT − zero-shot (pp)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Figure 2.  Chain-of-thought helps reasoning everywhere, but its effect on "
             "knowledge (MMLU) is model-specific", fontweight="bold", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "fig2_cot_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: the exchange rate on GSM8K (zero-shot vs CoT scaling) ──
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for k, fam in enumerate(families):
    ax = axes[k]
    sub = agg[(agg.family == fam) & (agg.benchmark == "gsm8k")]
    for s in ["zero_shot", "cot"]:
        ss = sub[sub.strategy == s].sort_values("params_b")
        ax.plot(ss.params_b, ss.acc_mean, marker=markers[s], color=colors[s],
                lw=2.6, ms=9, label=slabel[s])
    fmt_log(ax, sub.params_b.unique())
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.set_title(fam)
    ax.set_xlabel("parameters (B, log scale)")
    ax.legend(frameon=False)
axes[0].set_ylabel("GSM8K accuracy (%)")
fig.suptitle("Figure 3.  The prompt–parameter exchange rate on GSM8K: "
             "a small model with CoT beats a much larger one without it",
             fontweight="bold", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig3_exchange.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Wrote 3 figures to", OUT)
for p in sorted(OUT.glob("*.png")):
    print(" -", p.name)
