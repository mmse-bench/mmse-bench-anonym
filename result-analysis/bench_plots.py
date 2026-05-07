#!/usr/bin/env python3
"""
LoReS-Bench / MMSE-Bench Plotting Module  —  lores_bench_plots.py
==================================================================
35 existing plots + 6 new Block F cross-model consistency plots = 41 total.

New Block F plots (all new):
  PF1  label diversity histogram
  PF2  harm variance bubble (most disputed prompts)
  PF3  pairwise agreement heatmap
  PF4  per-language agreement bar (most contested languages)
  PF5  model agreement strip (sorted pairwise)
  PF6  disputed prompts label tile matrix

Usage
  from lores_bench_plots import plot_all
  plot_all(all_results, df_flat, output_dir="results/figures")
"""

from __future__ import annotations
from pathlib import Path
from math import pi

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from scipy.stats import gaussian_kde, pearsonr
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition  import PCA
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    import umap as umap_lib
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        180,
})
sns.set_theme(style="whitegrid", font_scale=1.0)

C_OPEN   = "#4C72B0"
C_CLOSED = "#C44E52"
PALETTE4 = ["#4C72B0","#DD8452","#55A868","#C44E52"]   # FR/PL/FC/IU
LABEL_SHORT = {"full_refusal":"FR","partial_refusal_with_leakage":"PL",
               "full_compliance":"FC","irrelevant_or_unintelligible":"IU"}
LABEL_COLORS = {"FR":"#4C72B0","PL":"#DD8452","FC":"#C44E52","IU":"#55A868"}
FAMILY_COLORS = {
    "openai":"#10a37f","openai_oss":"#34c78a","claude":"#d97706",
    "gemini":"#4285f4","llama":"#7c3aed","mistral":"#e11d48",
    "qwen":"#0891b2","gemma":"#059669","aya":"#b45309","other":"#6b7280",
}
DPI   = 200
FIG_W = 9
FIG_H = 5.5

def _save(fig, path: Path, name: str) -> None:
    fig.tight_layout()
    fp = path / f"{name}.png"
    fig.savefig(fp, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {fp.name}")

def _short(m: str) -> str:
    from lores_bench_analysis import _short_model
    return _short_model(m)

def _pct(ax, axis="y"):
    fmt = mticker.PercentFormatter(xmax=1)
    if axis == "y": ax.yaxis.set_major_formatter(fmt)
    else:           ax.xaxis.set_major_formatter(fmt)

def _radar(ax, categories, values_list, labels, colors, title="", fill_alpha=0.15):
    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)] + [0]
    ax.set_theta_offset(pi / 2); ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(categories, size=8)
    ax.set_yticklabels([]); ax.set_ylim(0, 1)
    ax.yaxis.grid(True, color="gray", alpha=0.3, linestyle="--")
    for vals, label, color in zip(values_list, labels, colors):
        v = list(vals) + [vals[0]]
        ax.plot(angles, v, color=color, linewidth=2, label=label)
        ax.fill(angles, v, color=color, alpha=fill_alpha)
    if title:
        ax.set_title(title, size=9, fontweight="bold", pad=14)


# ══════════════════════════════════════════════════════════════
# BLOCK A
# ══════════════════════════════════════════════════════════════

def plot_P1_label_donut(results, out):
    if "A1_label_distribution" not in results: return
    df = results["A1_label_distribution"].copy()
    df = df[~df.index.str.startswith("unsafe")]
    order = ["full_refusal","partial_refusal_with_leakage",
             "full_compliance","irrelevant_or_unintelligible"]
    short = {"full_refusal":"Full Refusal","partial_refusal_with_leakage":"Partial\nLeakage",
             "full_compliance":"Full\nCompliance","irrelevant_or_unintelligible":"IU /\nIncoherent"}
    df = df.reindex([l for l in order if l in df.index])
    vals = df["proportion"].values
    lbls = [f"{short[i]}\n{v*100:.1f}%" for i, v in zip(df.index, vals)]
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, _ = ax.pie(vals, labels=None, colors=PALETTE4,
                       wedgeprops={"linewidth":3,"edgecolor":"white"},
                       startangle=90, counterclock=False)
    ax.add_artist(plt.Circle((0,0), 0.55, color="white"))
    ax.text(0, 0, f"{vals[0]*100:.1f}%\nRefusal", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#333")
    ax.legend(wedges, lbls, loc="lower center", bbox_to_anchor=(0.5,-0.12),
              ncol=2, fontsize=9, frameon=False)
    ax.set_title("Response Label Distribution", fontweight="bold", fontsize=12, pad=8)
    _save(fig, out, "P1_label_distribution_donut")


def plot_P2_safety_lollipop(results, out):
    if "A2_core_safety_metrics" not in results: return
    df = results["A2_core_safety_metrics"].copy().dropna(subset=["value"])
    rename = {"ASR_full":"ASR_full","ASR_interp":"ASR_interp",
              "Leakage_Rate":"Leakage Rate","Full_Compliance_Rate":"Full Compliance",
              "IU_Rate":"IU Rate","High_Risk_Rate_(harm>0.8)":"High-Risk (>0.8)",
              "Safe_Rate_(FR)":"Safe (FR)","Leakage_Severity":"Leakage Severity",
              "Actionable_Steps_Rate":"Actionable Steps"}
    df.index = [rename.get(i,i) for i in df.index]
    colors = ["#C44E52" if any(k in i for k in ["ASR","Risk","Compliance","Leakage","Action"])
              else "#55A868" for i in df.index]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H+1))
    y = range(len(df))
    ax.hlines(y, 0, df["value"], colors=colors, linewidth=2.5, alpha=0.7)
    ax.scatter(df["value"], y, color=colors, s=80, zorder=5)
    for i, (idx, row) in enumerate(df.iterrows()):
        ax.text(row["value"]+0.008, i, f'{row["value"]*100:.1f}%',
                va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(list(y)); ax.set_yticklabels(df.index, fontsize=10)
    ax.set_xlabel("Rate"); _pct(ax,"x"); ax.set_xlim(0, df["value"].max()*1.3)
    ax.set_title("Core Safety Metrics (§4.5.1)", fontweight="bold", fontsize=12)
    _save(fig, out, "P2_core_safety_lollipop")


def plot_P3_harm_ridge(df_flat, out):
    needed = {"judge_harm_score","judge_label"}
    if not needed.issubset(df_flat.columns) or not _HAS_SCIPY: return
    df = df_flat.copy()
    df["label_s"] = df["judge_label"].map(LABEL_SHORT).fillna(df["judge_label"])
    order = ["FR","PL","FC","IU"]
    fig, axes = plt.subplots(len(order), 1, figsize=(FIG_W, 6),
                              sharex=True, gridspec_kw={"hspace":-0.3})
    for ax, label, color in zip(axes, order, PALETTE4):
        sub = df[df["label_s"]==label]["judge_harm_score"].dropna()
        if len(sub) < 5: continue
        xs = np.linspace(0, 1, 300)
        try: ys = gaussian_kde(sub, bw_method=0.15)(xs)
        except: ys = np.zeros_like(xs)
        ax.fill_between(xs, ys, alpha=0.6, color=color)
        ax.plot(xs, ys, color=color, linewidth=1.5)
        ax.set_yticks([]); ax.set_xlim(0,1)
        ax.spines["left"].set_visible(False); ax.spines["bottom"].set_visible(False)
        ax.text(-0.02, ys.max()*0.5, label, ha="right", va="center",
                fontsize=11, fontweight="bold", color=color, transform=ax.transData)
    axes[-1].set_xlabel("Harm Score"); axes[-1].spines["bottom"].set_visible(True)
    fig.suptitle("Harm Score Distribution by Response Label",
                 fontweight="bold", fontsize=12, y=1.01)
    _save(fig, out, "P3_harm_ridge_by_label")


def plot_P4_corr_heatmap(results, out):
    if "A5_correlation_matrix" not in results: return
    corr = results["A5_correlation_matrix"].copy()
    short = {"judge_harm_score":"Harm","judge_actionability_score":"Actionability",
             "judge_refusal_strength":"Refusal Strength"}
    corr = corr.rename(index=short, columns=short)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    mask = np.eye(len(corr), dtype=bool)
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, linewidths=1.5, linecolor="white",
                ax=ax, mask=mask, annot_kws={"size":13,"weight":"bold"},
                cbar_kws={"shrink":0.8,"label":"Pearson r"})
    for i in range(len(corr)):
        ax.text(i+0.5, i+0.5, "1.00", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#555")
    ax.set_title("Safety Score Correlation Matrix", fontweight="bold", fontsize=12)
    _save(fig, out, "P4_correlation_heatmap")


def plot_P5_hcas_rqs_hexbin(df_flat, out):
    needed = {"rqs","hcas","model_group"}
    if not needed.issubset(df_flat.columns): return
    df = df_flat.dropna(subset=["rqs","hcas"])
    groups = [g for g in ["open","closed"] if g in df["model_group"].values]
    if not groups: return
    fig, axes = plt.subplots(1, len(groups), figsize=(5*len(groups), 5),
                              sharey=True, sharex=True)
    if len(groups)==1: axes = [axes]
    for ax, grp in zip(axes, groups):
        sub = df[df["model_group"]==grp]
        hb  = ax.hexbin(sub["hcas"], sub["rqs"], gridsize=28,
                        cmap="Blues" if grp=="open" else "Reds",
                        mincnt=1, linewidths=0.2)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel("HCAS"); ax.set_ylabel("RQS")
        ax.set_title(f"{grp.capitalize()}-source", fontweight="bold")
        ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    fig.suptitle("HCAS vs RQS Density: Open vs Closed", fontweight="bold", fontsize=12)
    _save(fig, out, "P5_hcas_rqs_hexbin_open_closed")


def plot_PA6_cls_bubble(results, out):
    a7b = results.get("A7b_cls_per_language")
    c1  = results.get("C1_language_safety_full")
    if a7b is None or c1 is None: return
    merged = c1[["asr_full","count"]].join(a7b["cls_variance"], how="inner").dropna()
    if merged.empty: return
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    sc = ax.scatter(merged["cls_variance"], merged["asr_full"],
                    s=merged["count"]*0.8, c=merged["asr_full"],
                    cmap="RdYlGn_r", alpha=0.75, edgecolors="white",
                    linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="ASR_full")
    for lang in merged["asr_full"].nlargest(8).index:
        ax.annotate(lang, xy=(merged.loc[lang,"cls_variance"],merged.loc[lang,"asr_full"]),
                    xytext=(4,4), textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("CLS Variance"); ax.set_ylabel("ASR_full"); _pct(ax,"y")
    ax.set_title("Cross-Language Safety Consistency vs ASR", fontweight="bold", fontsize=11)
    _save(fig, out, "PA6_cls_bubble")


def plot_PA7_leakage_diverging(results, out):
    if "A8_leakage_dominant" not in results: return
    df = results["A8_leakage_dominant"].copy().dropna(subset=["LR","FCR"])
    df["lr_minus_fcr"] = (df["LR"] - df["FCR"]).round(4)
    df = df.sort_values("lr_minus_fcr", ascending=False)
    colors = ["#DD8452" if v>0 else "#4C72B0" for v in df["lr_minus_fcr"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.32)))
    ax.barh(df.index, df["lr_minus_fcr"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Leakage Rate − Full Compliance Rate")
    ax.set_title("Leakage-Dominant Languages (LR > FCR)", fontweight="bold", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color="#DD8452",label="Leakage-dominant"),
                        mpatches.Patch(color="#4C72B0",label="Compliance-dominant")],
              fontsize=9, framealpha=0.8)
    _save(fig, out, "PA7_leakage_diverging")


def plot_PA8_worst_case_bar(results, out):
    src = (results["A9_worst_case_harm"] if "A9_worst_case_harm" in results
           else results.get("C8_worst_case_harm_with_meta"))
    if src is None: return
    df = src.dropna(subset=["harm_max"]).head(20)
    colors = ["#C44E52" if v >= 1.0 else "#DD8452" for v in df["harm_max"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.35)))
    ax.barh(df.index, df["harm_max"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(1.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
    for lang, val in df["harm_max"].items():
        ax.text(val+0.005, lang, f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlabel("Maximum Observed Harm Score")
    ax.set_title("Worst-Case Harm per Language (T5/T10/T23)", fontweight="bold", fontsize=11)
    _save(fig, out, "PA8_worst_case_harm_bar")


def plot_PA9_leakage_severity(results, out):
    if "A10_leakage_severity_per_language" not in results: return
    df = results["A10_leakage_severity_per_language"].copy().head(20)
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.35)))
    ax.barh(df.index, df["leakage_severity"], color="#C44E52", edgecolor="white",
            height=0.65, alpha=0.85, label="Leakage Severity")
    ax.barh(df.index, df["leakage_harm"], color="#DD8452", edgecolor="white",
            height=0.35, alpha=0.7, label="Leakage Harm")
    ax.set_xlabel("Score [0–1]")
    ax.set_title("Leakage Severity per Language (§6.4)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "PA9_leakage_severity_bar")


def plot_PA10_language_match(results, out):
    """PA10 – Language match rate per language (new from real judge field)."""
    if "A11_language_match_rate" not in results: return
    df = results["A11_language_match_rate"].copy().dropna()
    df = df.sort_values("language_match_rate", ascending=True)
    colors = ["#C44E52" if v < 0.5 else "#55A868" for v in df["language_match_rate"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.32)))
    ax.barh(df.index, df["language_match_rate"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.9)
    for lang, val in df["language_match_rate"].items():
        ax.text(val+0.008, lang, f"{val*100:.0f}%", va="center", fontsize=8)
    ax.set_xlabel("Language Match Rate"); _pct(ax,"x")
    ax.set_title("Response Language Match Rate per Language\n(does model respond in prompt language?)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PA10_language_match_rate")


# ══════════════════════════════════════════════════════════════
# BLOCK B
# ══════════════════════════════════════════════════════════════

def plot_P6_translation_bucket(results, out):
    if "B2_translation_bucket_harm" not in results: return
    df = results["B2_translation_bucket_harm"].copy()
    cols = [c for c in ["harm_mean","asr_full","leakage","iu_rate"] if c in df.columns]
    pretty = {"harm_mean":"Mean Harm","asr_full":"ASR_full","leakage":"Leakage","iu_rate":"IU Rate"}
    x = np.arange(len(df)); w = 0.75/len(cols)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, col in enumerate(cols):
        ax.bar(x+(i-len(cols)/2+0.5)*w, df[col], width=w,
               label=pretty.get(col,col), color=sns.color_palette("Set2")[i],
               edgecolor="white", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylabel("Rate / Score"); _pct(ax,"y")
    ax.set_title("Safety Metrics by Translation Quality Bucket", fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "P6_translation_bucket")


def plot_P7_delta_safe_strip(results, out):
    src = (results["C2_delta_safe"] if "C2_delta_safe" in results
           else results.get("B3_tsc_delta_safe"))
    if src is None or "delta_safe" not in src.columns: return
    df = src[["delta_safe"]].dropna().sort_values("delta_safe", ascending=False)
    colors = ["#C44E52" if v>=0 else "#4C72B0" for v in df["delta_safe"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.30)))
    ax.scatter(df["delta_safe"], df.index, color=colors, s=55, zorder=5)
    for lang, row in df.iterrows():
        ax.hlines(lang, 0, row["delta_safe"], color="gray", alpha=0.4, linewidth=1)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Δ_safe = ASR_full(ℓ) − ASR_full(EN)"); _pct(ax,"x")
    ax.set_title("Safety Degradation vs English (Δ_safe)", fontweight="bold", fontsize=11)
    _save(fig, out, "P7_delta_safe_strip")


def plot_P8_cwur_slope(results, out):
    if "B4_cw_ur" not in results: return
    df = results["B4_cw_ur"].copy().dropna(subset=["asr_full","cw_ur"]).head(20)
    fig, ax = plt.subplots(figsize=(6, max(FIG_H, len(df)*0.38)))
    for lang, row in df.iterrows():
        color = "#C44E52" if row["asr_full"] > 0.5 else "#4C72B0"
        ax.plot([0,1],[row["asr_full"],row["cw_ur"]], color=color, alpha=0.6,
                linewidth=1.5, marker="o", markersize=4)
        ax.text(-0.03, row["asr_full"], lang, ha="right", va="center", fontsize=7.5)
        ax.text(1.03,  row["cw_ur"],   lang, ha="left",  va="center", fontsize=7.5)
    ax.set_xticks([0,1]); ax.set_xticklabels(["ASR_full","CW-UR"],
                                              fontsize=11, fontweight="bold")
    ax.set_ylabel("Rate"); _pct(ax,"y"); ax.set_xlim(-0.5, 1.5)
    ax.set_title("ASR_full → CW-UR Slope (§4.5.1)", fontweight="bold", fontsize=11)
    _save(fig, out, "P8_cwur_slope")


def plot_PB5_aegis_radar(results, out):
    if "E6_category_by_group" not in results: return
    e6 = results["E6_category_by_group"].copy().dropna()
    if "closed" not in e6.columns or "open" not in e6.columns: return
    cats = e6.index.tolist()[:8]
    if len(cats) < 3: return
    fig, ax = plt.subplots(1, 1, subplot_kw={"polar":True}, figsize=(6,6))
    _radar(ax, [c[:14] for c in cats],
           [[float(e6.loc[c,"closed"]) for c in cats],
            [float(e6.loc[c,"open"])   for c in cats]],
           ["Closed-source","Open-source"], [C_CLOSED,C_OPEN],
           title="Mean Harm by AEGIS Category\n(Open vs Closed)")
    ax.legend(loc="lower right", bbox_to_anchor=(1.35,0.0), fontsize=9)
    _save(fig, out, "PB5_aegis_radar_open_closed")


def plot_PB7_per_lang_tq_scatter(results, out):
    if "B7_per_language_tq_failure" not in results: return
    df = results["B7_per_language_tq_failure"].copy().dropna(subset=["trans_mean","failure_rate"])
    if df.empty: return
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    sc = ax.scatter(df["failure_rate"], df["trans_mean"],
                    c=df["trans_mean"], cmap="RdYlGn", vmin=0.45, vmax=1.0,
                    s=70, alpha=0.82, edgecolors="white", linewidths=0.5)
    plt.colorbar(sc, ax=ax, label="Mean BERTScore F1")
    if _HAS_SCIPY and len(df) > 3:
        m, b = np.polyfit(df["failure_rate"], df["trans_mean"], 1)
        xs = np.linspace(df["failure_rate"].min(), df["failure_rate"].max(), 100)
        r, _ = pearsonr(df["failure_rate"], df["trans_mean"])
        ax.plot(xs, m*xs+b, color="orange", linewidth=2, label=f"Trend (r²={r**2:.2f})")
        ax.legend(fontsize=9)
    ax.axvline(0.5,  color="gray",      linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axhline(0.75, color="steelblue", linestyle="--", linewidth=0.9, alpha=0.7)
    for lang in pd.concat([df["trans_mean"].nlargest(4), df["trans_mean"].nsmallest(4)]).index:
        ax.annotate(lang, xy=(df.loc[lang,"failure_rate"],df.loc[lang,"trans_mean"]),
                    xytext=(4,4), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("Failure Rate (IU rate)"); _pct(ax,"x")
    ax.set_ylabel("Mean BERTScore F1")
    ax.set_title("Translation Quality vs Failure Rate (Figure 7)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PB7_per_lang_tq_failure_scatter")


def plot_PB8_tq_resource_tier(results, out):
    if "B8_tq_by_resource_tier" not in results: return
    df = results["B8_tq_by_resource_tier"].copy().dropna(subset=["trans_mean"])
    order = [l for l in ["low","medium","high"] if l in df.index]
    df = df.reindex(order)
    colors = ["#C44E52","#F5A623","#55A868"][:len(df)]
    fig, ax = plt.subplots(figsize=(6, FIG_H))
    x = np.arange(len(df))
    bars = ax.bar(x, df["trans_mean"], color=colors, edgecolor="white", width=0.5, alpha=0.9)
    if "trans_std" in df.columns:
        ax.errorbar(x, df["trans_mean"], yerr=df["trans_std"],
                    fmt="none", color="black", capsize=5, linewidth=1.5)
    for bar, val in zip(bars, df["trans_mean"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{l.title()}" for l in df.index], fontsize=11)
    ax.set_ylabel("Mean BERTScore F1"); ax.set_ylim(0.5, 1.0)
    ax.set_title("Translation Quality by Resource Tier (Figure 12 top-left)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PB8_tq_by_resource_tier")


def plot_PB9_tq_prompt_length(results, out):
    if "B9_tq_vs_prompt_length" not in results: return
    df = results["B9_tq_vs_prompt_length"].copy().dropna(subset=["trans_mean"])
    if df.empty: return
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    x = np.arange(len(df))
    ax.fill_between(x, df["trans_mean"], alpha=0.2, color="steelblue")
    ax.plot(x, df["trans_mean"], marker="o", color="steelblue", linewidth=2, markersize=7)
    if "trans_std" in df.columns:
        ax.fill_between(x, df["trans_mean"]-df["trans_std"],
                        df["trans_mean"]+df["trans_std"], alpha=0.12, color="steelblue")
    for xi, val in zip(x, df["trans_mean"]):
        ax.text(xi, val+0.003, f"{val:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([str(i) for i in df.index], rotation=20, ha="right", fontsize=9)
    ax.set_xlabel("Prompt Length (chars)"); ax.set_ylabel("Mean BERTScore F1")
    ax.set_title("Translation Quality vs Prompt Length (Figure 12 top-right)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PB9_tq_vs_prompt_length")


def plot_PB10_tq_consistency_bubble(results, out):
    if "B11_language_quality_landscape" not in results: return
    df = results["B11_language_quality_landscape"].copy().dropna(
        subset=["trans_mean","failure_rate","trans_consistency"])
    if df.empty: return
    sizes = (df["trans_consistency"] / df["trans_consistency"].max() * 400 + 30).clip(30, 500)
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    sc = ax.scatter(df["failure_rate"], df["trans_mean"],
                    s=sizes, c=df["trans_mean"], cmap="RdYlGn",
                    vmin=0.45, vmax=1.0, alpha=0.78, edgecolors="white", linewidths=0.6)
    plt.colorbar(sc, ax=ax, label="Mean F1")
    for lang in pd.concat([df["trans_mean"].nlargest(4), df["trans_mean"].nsmallest(4)]).index:
        ax.annotate(lang, xy=(df.loc[lang,"failure_rate"],df.loc[lang,"trans_mean"]),
                    xytext=(4,4), textcoords="offset points", fontsize=7.5)
    ax.axvline(0.5,  color="gray",      linestyle=":", linewidth=1, alpha=0.7)
    ax.axhline(0.75, color="steelblue", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xlabel("Failure Rate"); _pct(ax,"x"); ax.set_ylabel("Mean BERTScore F1")
    ax.set_title("Language Quality Landscape (Figure 11)\nx=failure, y=F1, size∝1/σ",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PB10_language_quality_landscape_bubble")


def plot_PB11_best_worst_tq(results, out):
    if "B12_best_worst_10_languages_tq" not in results: return
    df = results["B12_best_worst_10_languages_tq"].copy().dropna(subset=["trans_mean"])
    top    = df[df["group"]=="top_10"].sort_values("trans_mean", ascending=True)
    bottom = df[df["group"]=="bottom_10"].sort_values("trans_mean", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W+2, FIG_H), sharey=False)
    for ax, sub, color, label in [(axes[0],bottom,"#C44E52","Bottom-10 (lowest TQ)"),
                                   (axes[1],top,  "#55A868","Top-10 (highest TQ)")]:
        bars = ax.barh(sub.index, sub["trans_mean"], color=color,
                       edgecolor="white", height=0.65, alpha=0.85)
        for bar, val in zip(bars, sub["trans_mean"]):
            ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=8)
        ax.set_xlabel("Mean BERTScore F1"); ax.set_xlim(0, 1.05)
        ax.axvline(0.75, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_title(label, fontweight="bold", fontsize=10)
    fig.suptitle("Best-10 vs Worst-10 Languages by TQ (Figure 15 bottom)",
                 fontweight="bold", fontsize=12)
    _save(fig, out, "PB11_best_worst_10_tq")


def plot_PB12_category_tq_dual(results, out):
    if "B13_category_tq_failure" not in results: return
    df = results["B13_category_tq_failure"].copy().dropna(subset=["trans_mean","failure_rate"])
    if df.empty: return
    df = df.sort_values("trans_mean", ascending=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W+3, max(FIG_H+1, len(df)*0.45)),
                                    sharey=True)
    def _color_bars(vals, cmap_name="RdYlGn"):
        norm = plt.Normalize(vals.min(), vals.max())
        return [plt.cm.get_cmap(cmap_name)(norm(v)) for v in vals]
    ax1.barh(df.index, df["trans_mean"], color=_color_bars(df["trans_mean"]),
             edgecolor="white", height=0.65)
    for lang, val in df["trans_mean"].items():
        ax1.text(val+0.002, lang, f"{val:.3f}", va="center", fontsize=8)
    ax1.axvline(df["trans_mean"].mean(), color="gray", linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Mean BERTScore F1"); ax1.set_xlim(0.5, 1.0)
    ax1.set_title("Translation Quality (F1)", fontweight="bold")
    ax2.barh(df.index, df["failure_rate"], color=_color_bars(df["failure_rate"],"RdYlGn_r"),
             edgecolor="white", height=0.65)
    for lang, val in df["failure_rate"].items():
        ax2.text(val+0.005, lang, f"{val*100:.1f}%", va="center", fontsize=8)
    ax2.axvline(df["failure_rate"].mean(), color="gray", linestyle="--", linewidth=0.9)
    ax2.set_xlabel("Failure Rate"); _pct(ax2,"x"); ax2.set_title("Failure Rate", fontweight="bold")
    fig.suptitle("Harm Category — Translation Quality & Failure Rate (Figure 8/10, Table 17)",
                 fontweight="bold", fontsize=12)
    _save(fig, out, "PB12_category_tq_failure_dual")


def plot_PB13_resource_category_tq_heatmap(results, out):
    if "B14_resource_category_tq_heatmap" not in results: return
    pivot = results["B14_resource_category_tq_heatmap"].copy()
    disp = pivot.drop(index="delta_med_minus_low", errors="ignore")
    disp = disp.drop(columns="delta_med_minus_low", errors="ignore").astype(float)
    if disp.empty: return
    fig, ax = plt.subplots(figsize=(8, max(4, len(disp.columns)*0.55)))
    sns.heatmap(disp.T, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.60, vmax=0.90,
                linewidths=0.8, linecolor="white", ax=ax,
                annot_kws={"size":9,"weight":"bold"},
                cbar_kws={"label":"Mean BERTScore F1","shrink":0.6})
    ax.set_xlabel("Resource Tier"); ax.set_ylabel("Harm Category")
    ax.set_title("BERTScore F1 by Harm Category × Resource Tier (Figure 9, Table 18)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PB13_resource_category_tq_heatmap")


# ══════════════════════════════════════════════════════════════
# BLOCK C
# ══════════════════════════════════════════════════════════════

def plot_P9_language_harm_bar(results, out):
    if "C1_language_safety_full" not in results: return
    df = results["C1_language_safety_full"].copy().dropna(subset=["harm"])
    combo = pd.concat([df["harm"].nlargest(12),df["harm"].nsmallest(8)]).drop_duplicates().sort_values()
    colors = ["#C44E52" if v>=combo.median() else "#4C72B0" for v in combo]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(combo)*0.38)))
    ax.barh(combo.index, combo.values, color=colors, edgecolor="white", height=0.65)
    ax.axvline(combo.median(), color="gray", linestyle="--", linewidth=0.9, label="median")
    for lang, val in combo.items():
        ax.text(val+0.005, lang, f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Harm Score")
    ax.set_title("Language-Level Harm Extremes (Top-12 / Bottom-8)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "P9_language_harm_extremes")


def plot_P10_resource_grouped(results, out):
    if "C3_resource_level" not in results: return
    df = results["C3_resource_level"].copy()
    cols = [c for c in ["asr_full","asr_interp","iu_rate","asr_gap"] if c in df.columns]
    pretty = {"asr_full":"ASR_full","asr_interp":"ASR_interp",
              "iu_rate":"IU Rate","asr_gap":"Δ(interp−full)"}
    x = np.arange(len(df)); w = 0.75/len(cols)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, col in enumerate(cols):
        ax.bar(x+(i-len(cols)/2+0.5)*w, df[col], width=w,
               label=pretty[col], color=sns.color_palette("Set2")[i], edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(df.index, fontsize=10)
    ax.set_ylabel("Rate"); _pct(ax,"y")
    ax.set_title("ASR Gap by Resource Level (IU-masking diagnostic)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "P10_resource_level_grouped")


def plot_P11_joshi_heatmap(results, out):
    if "C4_joshi_class" not in results: return
    df = results["C4_joshi_class"].copy()
    cols = [c for c in ["harm","asr_full","leakage","iu_rate"] if c in df.columns]
    fig, ax = plt.subplots(figsize=(FIG_W, 3.5))
    sns.heatmap(df[cols].astype(float).T, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.8, linecolor="white", ax=ax,
                cbar_kws={"shrink":0.6,"label":"Rate / Score"},
                annot_kws={"size":11,"weight":"bold"})
    ax.set_xlabel("Joshi Class  (1=fewest → 5=most resources)")
    ax.set_yticklabels([c.replace("_"," ").title() for c in cols], fontsize=10)
    ax.set_title("Safety Heatmap by Joshi Resource Class", fontweight="bold", fontsize=11)
    _save(fig, out, "P11_joshi_heatmap")


def plot_P12_pareto_scatter(results, out):
    if "C1_language_safety_full" not in results: return
    df = results["C1_language_safety_full"].copy().dropna(subset=["harm","actionability"])
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    sc = ax.scatter(df["actionability"], df["harm"],
                    c=df["asr_full"] if "asr_full" in df.columns else df["harm"],
                    cmap="RdYlGn_r", s=df.get("count",60)*0.5+25,
                    alpha=0.78, edgecolors="white", linewidths=0.5, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="ASR_full")
    for lang in pd.concat([df["harm"].nlargest(6), df["harm"].nsmallest(5)]).index:
        ax.annotate(lang, xy=(df.loc[lang,"actionability"],df.loc[lang,"harm"]),
                    xytext=(4,4), textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Mean Actionability Score"); ax.set_ylabel("Mean Harm Score")
    ax.set_title("Language-Level Harm vs Actionability (Pareto / G6)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "P12_pareto_harm_actionability")


def plot_PC5_family_radar(results, out):
    if "C5_language_family" not in results: return
    df = results["C5_language_family"].copy()
    cats = [c for c in ["harm","asr_full","leakage","rqs"] if c in df.columns]
    if len(cats) < 3 or len(df) < 2: return
    families = df.index.tolist()[:6]
    ncols = min(3, len(families)); nrows = int(np.ceil(len(families)/ncols))
    fig = plt.figure(figsize=(5*ncols, 4.5*nrows))
    pal = sns.color_palette("tab10", len(families))
    for i, (fam, color) in enumerate(zip(families, pal)):
        ax = fig.add_subplot(nrows, ncols, i+1, polar=True)
        vals = [float(df.loc[fam,c]) if c in df.columns else 0 for c in cats]
        _radar(ax, [c.replace("_"," ").title() for c in cats],
               [vals], [fam], [color], title=fam, fill_alpha=0.25)
    fig.suptitle("Safety Profile per Language Family", fontweight="bold", fontsize=13, y=1.01)
    _save(fig, out, "PC5_family_radar_grid")


def plot_PC_script_bubble(results, out):
    if "C6_script_safety" not in results: return
    df = results["C6_script_safety"].copy().dropna(subset=["harm"])
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    sc = ax.scatter(range(len(df)), df["harm"], s=df.get("count",50)*0.5+40,
                    c=df.get("iu_rate",df["harm"]), cmap="YlOrRd",
                    alpha=0.82, edgecolors="white", linewidths=0.6, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label="IU Rate")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Harm Score")
    ax.set_title("Harm by Writing Script (bubble size = count)", fontweight="bold", fontsize=11)
    _save(fig, out, "PC_script_bubble")


def plot_PC9_resource_category_harm_heatmap(results, out):
    if "C9_resource_category_harm_heatmap" not in results: return
    pivot = results["C9_resource_category_harm_heatmap"].copy().astype(float)
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot.columns)*0.5)))
    sns.heatmap(pivot.T, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.8, linecolor="white", ax=ax,
                annot_kws={"size":9,"weight":"bold"},
                cbar_kws={"label":"Mean Harm Score","shrink":0.6})
    ax.set_xlabel("Resource Tier"); ax.set_ylabel("Harm Category")
    ax.set_title("Mean Harm by Harm Category × Resource Tier",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PC9_resource_category_harm_heatmap")


# ══════════════════════════════════════════════════════════════
# BLOCK D
# ══════════════════════════════════════════════════════════════

def plot_P13_model_dot_matrix(results, out):
    if "D1_model_safety" not in results: return
    df = results["D1_model_safety"].copy()
    cols = [c for c in ["harm","asr_full","asr_interp","leakage","iu_rate","rqs"]
            if c in df.columns]
    e2 = results.get("E2_per_model_profile")
    def _grp(idx):
        if e2 is not None and idx in e2.index and "model_group" in e2.columns:
            return str(e2.loc[idx,"model_group"])
        return "closed" if any(k in str(idx).lower()
                               for k in ["claude","gemini","gpt-5","gpt-4"]) else "open"
    row_colors = [C_CLOSED if _grp(i)=="closed" else C_OPEN for i in df.index]
    fig, ax = plt.subplots(figsize=(max(FIG_W,len(cols)*1.4), max(FIG_H,len(df)*0.7)))
    for j, col in enumerate(cols):
        for i, (idx, row) in enumerate(df.iterrows()):
            val = float(row[col]) if not pd.isna(row[col]) else 0
            ax.scatter(j, i, s=val*600+20, c=row_colors[i], alpha=0.75,
                       edgecolors="white", linewidths=0.5)
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_"," ").upper() for c in cols], fontsize=9)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([_short(m) for m in df.index], fontsize=9)
    ax.legend(handles=[mpatches.Patch(color=C_CLOSED,label="Closed"),
                        mpatches.Patch(color=C_OPEN,  label="Open")],
              fontsize=9, loc="lower right", framealpha=0.8)
    ax.set_title("Model Safety Dot Matrix (dot size ∝ metric value)",
                 fontweight="bold", fontsize=11)
    ax.set_xlim(-0.7,len(cols)-0.3); ax.set_ylim(-0.7,len(df)-0.3)
    ax.invert_yaxis(); ax.grid(True, alpha=0.2)
    _save(fig, out, "P13_model_dot_matrix")


def plot_P14_model_lang_heatmap(results, out):
    if "D4_model_language_harm_pivot" not in results: return
    pivot = results["D4_model_language_harm_pivot"].copy()
    if pivot.shape[1] > 28:
        pivot = pivot[pivot.var(axis=0).sort_values(ascending=False).head(28).index]
    if pivot.empty: return
    fig, ax = plt.subplots(figsize=(max(FIG_W+3,len(pivot.columns)*0.38),
                                     max(4,len(pivot)*0.65)))
    sns.heatmap(pivot, cmap="YlOrRd", vmin=0, vmax=1,
                linewidths=0.3, linecolor="white",
                annot=(pivot.shape[1]<=14), fmt=".2f", annot_kws={"size":7},
                cbar_kws={"label":"Mean Harm Score","shrink":0.6}, ax=ax)
    ax.set_xlabel("Language"); ax.set_ylabel("Model")
    ax.set_yticklabels([_short(m) for m in pivot.index], rotation=0, fontsize=8)
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7.5)
    ax.set_title("Model × Language Harm Heatmap", fontweight="bold", fontsize=11)
    _save(fig, out, "P14_model_language_heatmap")


def plot_PD3_family_grouped(results, out):
    if "E5_family_safety_profile" not in results: return
    df = results["E5_family_safety_profile"].copy().reset_index()
    if "model_group" not in df.columns: return
    cols = [c for c in ["harm","asr_full","leakage","rqs"] if c in df.columns]
    families = df["model_family"].unique() if "model_family" in df.columns else df.index.unique()
    x = np.arange(len(families)); w = 0.75/len(cols)
    fig, ax = plt.subplots(figsize=(max(FIG_W,len(families)*1.2), FIG_H))
    pal = sns.color_palette("Set2", len(cols))
    for i, col in enumerate(cols):
        vals = [float(df[df["model_family"]==fam][col].mean())
                if "model_family" in df.columns and len(df[df["model_family"]==fam])>0
                else 0 for fam in families]
        ax.bar(x+(i-len(cols)/2+0.5)*w, vals, width=w,
               label=col.replace("_"," "), color=pal[i], edgecolor="white", alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(families, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Rate / Score"); _pct(ax,"y")
    ax.set_title("Safety Metrics by Model Family", fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "PD3_family_safety_grouped")


def plot_PD_ocsg_diverging(results, out):
    if "E3_ocsg_per_language" not in results: return
    df = results["E3_ocsg_per_language"].copy()
    if "ocsg_per_lang" not in df.columns: return
    df = df[["ocsg_per_lang"]].dropna().sort_values("ocsg_per_lang", ascending=False)
    colors = ["#C44E52" if v>=0 else "#4C72B0" for v in df["ocsg_per_lang"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H,len(df)*0.30)))
    ax.barh(df.index, df["ocsg_per_lang"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(0, color="black", linewidth=1, linestyle="--"); _pct(ax,"x")
    ax.set_xlabel("OCSG_per_lang = ASR_closed − ASR_open")
    ax.set_title("Per-Language Open vs Closed Safety Gap", fontweight="bold", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color="#C44E52",label="Closed worse"),
                        mpatches.Patch(color="#4C72B0",label="Open worse")],
              fontsize=9, framealpha=0.8)
    _save(fig, out, "PD_ocsg_per_language_diverging")


def plot_PD7_response_length(results, out):
    """PD7 – Response length distribution by model (new)."""
    if "D7b_response_length_per_model" not in results: return
    df = results["D7b_response_length_per_model"].copy().dropna(subset=["resp_len_mean"])
    df = df.sort_values("resp_len_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.55)))
    colors = [C_CLOSED if any(k in _short(str(m)).lower()
                               for k in ["claude","gemini","gpt-5","mini"])
              else C_OPEN for m in df.index]
    ax.barh([_short(str(m)) for m in df.index], df["resp_len_mean"],
            xerr=df.get("resp_len_std"), color=colors, edgecolor="white",
            height=0.65, alpha=0.85, capsize=4)
    ax.set_xlabel("Mean Response Length (chars)")
    ax.set_title("Response Length per Model  (coloured by open/closed)",
                 fontweight="bold", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=C_CLOSED,label="Closed"),
                        mpatches.Patch(color=C_OPEN,  label="Open")],
              fontsize=9, framealpha=0.8)
    _save(fig, out, "PD7_response_length_per_model")


def plot_PD10_latency_per_model(results, out):
    """PD10 – Latency comparison per model (new)."""
    if "D10_latency_per_model" not in results: return
    df = results["D10_latency_per_model"].copy().dropna(subset=["latency_mean"])
    df = df.sort_values("latency_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.55)))
    colors = [C_CLOSED if any(k in _short(str(m)).lower()
                               for k in ["claude","gemini","gpt-5","mini"])
              else C_OPEN for m in df.index]
    ax.barh([_short(str(m)) for m in df.index], df["latency_mean"],
            xerr=df.get("latency_std"), color=colors, edgecolor="white",
            height=0.65, alpha=0.85, capsize=4)
    ax.set_xlabel("Mean Latency (sec / sample)")
    ax.set_title("Inference Latency per Model", fontweight="bold", fontsize=11)
    ax.legend(handles=[mpatches.Patch(color=C_CLOSED,label="Closed"),
                        mpatches.Patch(color=C_OPEN,  label="Open")],
              fontsize=9, framealpha=0.8)
    _save(fig, out, "PD10_latency_per_model")


# ══════════════════════════════════════════════════════════════
# BLOCK E
# ══════════════════════════════════════════════════════════════

def plot_PE1_multi_radar(results, out):
    e1 = results.get("E1_group_safety_profile")
    e2 = results.get("E2_per_model_profile")
    if e1 is None: return
    radar_cols = [c for c in ["asr_full","leakage","fcr","iu_rate","harm","actionability","rqs"]
                  if c in e1.columns]
    if len(radar_cols) < 4: return
    groups   = [g for g in ["open","closed"] if g in e1.index]
    n_models = min(4, len(e2)) if e2 is not None else 0
    ncols = 1 + n_models
    fig = plt.figure(figsize=(4.5*ncols, 5))
    ax0 = fig.add_subplot(1, ncols, 1, polar=True)
    _radar(ax0, [c.replace("_"," ").title() for c in radar_cols],
           [[float(e1.loc[g,c]) if c in e1.columns else 0 for c in radar_cols] for g in groups],
           groups, [C_OPEN if g=="open" else C_CLOSED for g in groups],
           title="Open vs Closed\n(Aggregate)", fill_alpha=0.20)
    ax0.legend(loc="lower right", bbox_to_anchor=(1.4,-0.1), fontsize=8)
    if e2 is not None and n_models > 0:
        for k, (model, row) in enumerate(e2.head(n_models).iterrows()):
            ax = fig.add_subplot(1, ncols, k+2, polar=True)
            grp   = str(row.get("model_group","open"))
            vals  = [float(row[c]) if c in row.index and not pd.isna(row[c]) else 0
                     for c in radar_cols]
            _radar(ax, [c.replace("_"," ").title() for c in radar_cols],
                   [vals], [_short(model)],
                   [C_CLOSED if grp=="closed" else C_OPEN],
                   title=_short(model)[:18], fill_alpha=0.25)
    fig.suptitle("Multi-Radar Safety Profiles: Open vs Closed Models",
                 fontweight="bold", fontsize=13, y=1.04)
    _save(fig, out, "PE1_multi_radar_open_closed")


def plot_PE2_umap_pca(results, df_flat, out):
    if "E8_model_feature_vectors" not in results or not _HAS_SKLEARN: return
    e8  = results["E8_model_feature_vectors"].copy().dropna()
    e2  = results.get("E2_per_model_profile")
    if len(e8) < 3: return
    X = StandardScaler().fit_transform(e8.values.astype(float))
    if _HAS_UMAP and len(e8) >= 4:
        n_nb = min(len(e8)-1, 3)
        emb    = umap_lib.UMAP(n_components=2, random_state=42, n_neighbors=n_nb,
                                min_dist=0.3).fit_transform(X)
        method = "UMAP"
    else:
        emb = PCA(n_components=2, random_state=42).fit_transform(X); method = "PCA"
    emb_df = pd.DataFrame(emb, index=e8.index, columns=["x","y"])
    if e2 is not None:
        for col in ["model_group","model_family"]:
            if col in e2.columns:
                emb_df[col] = e2[col].reindex(emb_df.index)
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    for _, row in emb_df.iterrows():
        grp    = str(row.get("model_group","open"))
        fam    = str(row.get("model_family","other"))
        color  = FAMILY_COLORS.get(fam, "#999999")
        marker = "D" if grp=="closed" else "o"
        ax.scatter(row["x"], row["y"], c=color, marker=marker,
                   s=180, edgecolors="white", linewidths=1.2, zorder=5)
        ax.annotate(_short(row.name), xy=(row["x"],row["y"]),
                    xytext=(5,5), textcoords="offset points",
                    fontsize=8, fontweight="bold", color="#333")
    ax.legend(handles=[
        Line2D([0],[0],marker="D",color="w",markerfacecolor="gray",markersize=9,label="Closed"),
        Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markersize=9,label="Open"),
    ] + [mpatches.Patch(color=v,label=k)
         for k,v in FAMILY_COLORS.items()
         if k in emb_df.get("model_family",pd.Series()).values],
    fontsize=8, framealpha=0.85, loc="upper left", ncol=2)
    ax.set_xlabel(f"{method} dim-1"); ax.set_ylabel(f"{method} dim-2")
    ax.set_title(f"{method} Embedding of Model Safety Profiles\n(◆=closed, ●=open; colour=family)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PE2_model_embedding_umap_pca")


def plot_PE3_ocsg_strip(results, out):
    if "E3_ocsg_per_language" not in results: return
    df = results["E3_ocsg_per_language"].copy()
    if "closed" not in df.columns or "open" not in df.columns: return
    df = df.dropna(subset=["closed","open"]).sort_values("ocsg_per_lang", ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(6, max(FIG_H,len(df)*0.38)))
    for i, (lang, row) in enumerate(df.iterrows()):
        c_val = float(row["closed"]); o_val = float(row["open"])
        color = "#C44E52" if c_val >= o_val else "#4C72B0"
        ax.plot([o_val,c_val],[i,i], color=color, linewidth=2, alpha=0.7)
        ax.scatter([o_val],[i], color=C_OPEN,   s=60, zorder=5, marker="o")
        ax.scatter([c_val],[i], color=C_CLOSED, s=60, zorder=5, marker="D")
    ax.set_yticks(range(len(df))); ax.set_yticklabels(df.index, fontsize=9)
    ax.set_xlabel("ASR_full"); _pct(ax,"x")
    ax.set_title("ASR_full: Open (●) vs Closed (◆) per Language",
                 fontweight="bold", fontsize=11)
    ax.legend(handles=[
        Line2D([0],[0],marker="o",color="w",markerfacecolor=C_OPEN,  markersize=9,label="Open"),
        Line2D([0],[0],marker="D",color="w",markerfacecolor=C_CLOSED,markersize=9,label="Closed"),
    ], fontsize=9, framealpha=0.8)
    _save(fig, out, "PE3_ocsg_language_strip")


def plot_PE4_label_mix_alluvial(results, out):
    if "E4_label_mix_by_group" not in results: return
    df = results["E4_label_mix_by_group"].copy()
    order = ["full_refusal","partial_refusal_with_leakage",
             "full_compliance","irrelevant_or_unintelligible"]
    cols  = [c for c in order if c in df.columns]
    short = {"full_refusal":"Full Refusal","partial_refusal_with_leakage":"Partial Leakage",
             "full_compliance":"Full Compliance","irrelevant_or_unintelligible":"IU"}
    if not cols: return
    fig, ax = plt.subplots(figsize=(6, FIG_H))
    bottoms = np.zeros(len(df)); x = np.arange(len(df))
    for col, color in zip(cols, PALETTE4):
        vals = df[col].values if col in df.columns else np.zeros(len(df))
        ax.bar(x, vals, bottom=bottoms, color=color, edgecolor="white",
               width=0.5, label=short.get(col,col))
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(xi, b+v/2, f"{v*100:.0f}%", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([g.capitalize()+"\nSource" for g in df.index],
                        fontsize=11, fontweight="bold")
    ax.set_ylabel("Proportion"); _pct(ax,"y"); ax.set_ylim(0,1.05)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5,-0.22), ncol=2, fontsize=9, frameon=False)
    ax.set_title("Response Label Mix: Open vs Closed (100% stacked)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PE4_label_mix_alluvial")


def plot_PE5_bump_chart(results, out):
    if "E3_ocsg_per_language" not in results: return
    df = results["E3_ocsg_per_language"].copy()
    if "closed" not in df.columns or "open" not in df.columns: return
    df = df.dropna(subset=["closed","open"])
    if len(df) < 4: return
    df = df.head(20)
    rank_open   = df["open"].rank(ascending=True).astype(int)
    rank_closed = df["closed"].rank(ascending=True).astype(int)
    fig, ax = plt.subplots(figsize=(7, max(FIG_H+1, len(df)*0.38)))
    cmap = plt.cm.get_cmap("tab20", len(df))
    for i, (lang, _) in enumerate(df.iterrows()):
        ro = rank_open[lang]; rc = rank_closed[lang]
        color = cmap(i)
        ax.plot([0,1],[ro,rc], color=color, linewidth=2.2, alpha=0.8)
        ax.scatter([0],[ro], color=color, s=60, zorder=5)
        ax.scatter([1],[rc], color=color, s=60, zorder=5)
        ax.text(-0.05, ro, lang, ha="right", va="center", fontsize=8, color=color)
        ax.text(1.05,  rc, lang, ha="left",  va="center", fontsize=8, color=color)
    ax.set_xticks([0,1])
    ax.set_xticklabels(["Open-source","Closed-source"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Safety Rank (lower = safer)"); ax.invert_yaxis(); ax.set_xlim(-0.6,1.6)
    ax.set_title("Language Safety Rank Bump Chart (Open vs Closed)",
                 fontweight="bold", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out, "PE5_bump_chart_open_closed")


# ══════════════════════════════════════════════════════════════
# BLOCK F  —  CROSS-MODEL CONSISTENCY  (all new)
# ══════════════════════════════════════════════════════════════

def plot_PF1_label_diversity_hist(results, out):
    """PF1 – Histogram of label diversity across models per prompt."""
    if "F1_cross_model_label_diversity" not in results: return
    df = results["F1_cross_model_label_diversity"].copy()
    df.index = df.index.astype(str)
    fig, ax = plt.subplots(figsize=(6, FIG_H))
    colors = ["#55A868" if i=="1" else ("#F5A623" if i=="2" else "#C44E52")
              for i in df.index]
    bars = ax.bar(df.index, df["prompt_count"], color=colors, edgecolor="white", width=0.6)
    ax2 = ax.twinx()
    ax2.plot(df.index, df["proportion"]*100, color="black", marker="o",
             linewidth=2, markersize=6, label="% of prompts")
    for bar, val in zip(bars, df["prompt_count"]):
        ax.text(bar.get_x()+bar.get_width()/2, val+10, f"{val:,}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xlabel("Number of Distinct Labels Assigned (across models)")
    ax.set_ylabel("Prompt Count")
    ax2.set_ylabel("% of Prompts")
    ax.set_title("Cross-Model Label Diversity per Prompt\n1=full agreement, 2+=disagreement",
                 fontweight="bold", fontsize=11)
    ax2.legend(fontsize=9, loc="upper right")
    _save(fig, out, "PF1_label_diversity_histogram")


def plot_PF2_harm_variance_bubble(results, out):
    """PF2 – Bubble chart of most disputed prompts by harm variance."""
    if "F2b_most_disputed_prompts" not in results: return
    df = results["F2b_most_disputed_prompts"].copy().dropna(subset=["harm_mean","harm_variance"])
    if df.empty: return
    fig, ax = plt.subplots(figsize=(FIG_W+1, FIG_H+1))
    sc = ax.scatter(df["harm_mean"], df["harm_variance"],
                    s=150, c=df["harm_variance"], cmap="OrRd",
                    alpha=0.82, edgecolors="white", linewidths=0.8, vmin=0)
    plt.colorbar(sc, ax=ax, label="Harm Variance across models")
    for prompt_id, row in df.iterrows():
        lang = str(row.get("language",""))
        ax.annotate(f"{str(prompt_id)[:8]}…\n{lang}",
                    xy=(row["harm_mean"], row["harm_variance"]),
                    xytext=(4,4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Mean Harm Score"); ax.set_ylabel("Harm Variance across Models")
    ax.set_title("Most Disputed Prompts (highest cross-model harm variance)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PF2_disputed_prompts_harm_variance_bubble")


def plot_PF3_pairwise_agreement_heatmap(results, out):
    """PF3 – Pairwise model agreement heatmap."""
    if "F3b_pairwise_agreement_matrix" not in results: return
    mat = results["F3b_pairwise_agreement_matrix"].copy().astype(float)
    if mat.empty: return
    fig, ax = plt.subplots(figsize=(max(5, len(mat)*1.1), max(4, len(mat)*0.9)))
    mask = np.zeros_like(mat, dtype=bool)
    np.fill_diagonal(mask, False)
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.5, vmax=1.0, linewidths=1.0, linecolor="white",
                ax=ax, annot_kws={"size":11,"weight":"bold"},
                cbar_kws={"label":"Agreement Rate (safe/unsafe)","shrink":0.7})
    ax.set_title("Pairwise Model Agreement Rate\n(safe vs unsafe binary; diagonal=1.0)",
                 fontweight="bold", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    _save(fig, out, "PF3_pairwise_agreement_heatmap")


def plot_PF4_per_language_agreement(results, out):
    """PF4 – Per-language mean pairwise agreement bar (most contested languages)."""
    if "F4_cross_model_agreement_per_language" not in results: return
    df = results["F4_cross_model_agreement_per_language"].copy().dropna(
        subset=["mean_pairwise_agreement"])
    df = df.sort_values("mean_pairwise_agreement", ascending=True).head(25)
    colors = ["#C44E52" if v < 0.6 else ("#F5A623" if v < 0.75 else "#55A868")
              for v in df["mean_pairwise_agreement"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.38)))
    ax.barh(df.index, df["mean_pairwise_agreement"], color=colors,
            edgecolor="white", height=0.65)
    ax.axvline(0.75, color="gray", linestyle="--", linewidth=0.9, label="0.75 threshold")
    for lang, val in df["mean_pairwise_agreement"].items():
        ax.text(val+0.005, lang, f"{val:.3f}", va="center", fontsize=8)
    ax.set_xlabel("Mean Pairwise Agreement Rate"); _pct(ax,"x")
    ax.set_title("Most Contested Languages (lowest cross-model agreement)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    _save(fig, out, "PF4_per_language_agreement_bar")


def plot_PF5_model_agreement_strip(results, out):
    """PF5 – Sorted pairwise agreement strip — all model pairs ranked."""
    if "F3_pairwise_model_agreement" not in results: return
    df = results["F3_pairwise_model_agreement"].copy().dropna(subset=["agreement_rate"])
    df = df.sort_values("agreement_rate", ascending=True)
    df["pair"] = df["model_A"] + "\n× " + df["model_B"]
    colors = ["#C44E52" if v < 0.6 else ("#F5A623" if v < 0.75 else "#55A868")
              for v in df["agreement_rate"]]
    fig, ax = plt.subplots(figsize=(FIG_W, max(FIG_H, len(df)*0.55)))
    ax.barh(df["pair"], df["agreement_rate"], color=colors, edgecolor="white", height=0.65)
    ax.axvline(0.75, color="gray", linestyle="--", linewidth=0.9)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["agreement_rate"]+0.005, i, f"{row['agreement_rate']*100:.1f}%",
                va="center", fontsize=8)
    ax.set_xlabel("Agreement Rate (safe/unsafe binary)"); _pct(ax,"x")
    ax.set_title("Pairwise Model Agreement  (sorted lowest → highest)",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PF5_pairwise_agreement_strip")


def plot_PF6_disputed_label_tile(results, out):
    """PF6 – Tile matrix: what label each model assigned to most disputed prompts."""
    if "F5_disputed_prompts_label_table" not in results: return
    df = results["F5_disputed_prompts_label_table"].copy()
    model_cols = [c for c in df.columns if c != "language"]
    if not model_cols: return

    # encode labels as numbers for heatmap
    label_enc = {"full_refusal":0, "partial_refusal_with_leakage":1,
                 "full_compliance":2, "irrelevant_or_unintelligible":3}
    enc = df[model_cols].applymap(lambda x: label_enc.get(str(x), np.nan))

    fig, ax = plt.subplots(figsize=(max(7, len(model_cols)*1.5),
                                     max(4, len(df)*0.42)))
    cmap_custom = matplotlib.colors.ListedColormap(PALETTE4)
    im = ax.imshow(enc.values, aspect="auto", cmap=cmap_custom, vmin=0, vmax=3)
    ax.set_xticks(range(len(model_cols)))
    ax.set_xticklabels(model_cols, rotation=30, ha="right", fontsize=9)
    row_labels = [f"{str(idx)[:10]} ({df.loc[idx,'language']})"
                  if "language" in df.columns else str(idx)[:14]
                  for idx in df.index]
    ax.set_yticks(range(len(df))); ax.set_yticklabels(row_labels, fontsize=8)
    # annotate
    for i in range(len(df)):
        for j, col in enumerate(model_cols):
            lbl = str(df.iloc[i][col])
            short = LABEL_SHORT.get(lbl, "?")
            ax.text(j, i, short, ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white")
    cbar = plt.colorbar(im, ax=ax, ticks=[0,1,2,3], shrink=0.5)
    cbar.set_ticklabels(["FR","PL","FC","IU"])
    ax.set_title("Most Disputed Prompts — Label Assigned by Each Model",
                 fontweight="bold", fontsize=11)
    _save(fig, out, "PF6_disputed_prompts_label_tile")


# ══════════════════════════════════════════════════════════════
# MASTER RUNNER
# ══════════════════════════════════════════════════════════════

def plot_all(all_results: dict, df_flat: pd.DataFrame,
             output_dir: str = "results/figures") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n── Generating plots → {out.resolve()} ──────────────────")

    import matplotlib as _mpl
    # make applymap work across pandas versions
    if not hasattr(pd.DataFrame, 'applymap'):
        pd.DataFrame.applymap = pd.DataFrame.map

    # Block A
    plot_P1_label_donut(all_results, out)
    plot_P2_safety_lollipop(all_results, out)
    plot_P3_harm_ridge(df_flat, out)
    plot_P4_corr_heatmap(all_results, out)
    plot_P5_hcas_rqs_hexbin(df_flat, out)
    plot_PA6_cls_bubble(all_results, out)
    plot_PA7_leakage_diverging(all_results, out)
    plot_PA8_worst_case_bar(all_results, out)
    plot_PA9_leakage_severity(all_results, out)
    plot_PA10_language_match(all_results, out)

    # Block B
    plot_P6_translation_bucket(all_results, out)
    plot_P7_delta_safe_strip(all_results, out)
    plot_P8_cwur_slope(all_results, out)
    plot_PB5_aegis_radar(all_results, out)
    plot_PB7_per_lang_tq_scatter(all_results, out)
    plot_PB8_tq_resource_tier(all_results, out)
    plot_PB9_tq_prompt_length(all_results, out)
    plot_PB10_tq_consistency_bubble(all_results, out)
    plot_PB11_best_worst_tq(all_results, out)
    plot_PB12_category_tq_dual(all_results, out)
    plot_PB13_resource_category_tq_heatmap(all_results, out)

    # Block C
    plot_P9_language_harm_bar(all_results, out)
    plot_P10_resource_grouped(all_results, out)
    plot_P11_joshi_heatmap(all_results, out)
    plot_P12_pareto_scatter(all_results, out)
    plot_PC5_family_radar(all_results, out)
    plot_PC_script_bubble(all_results, out)
    plot_PC9_resource_category_harm_heatmap(all_results, out)

    # Block D
    plot_P13_model_dot_matrix(all_results, out)
    plot_P14_model_lang_heatmap(all_results, out)
    plot_PD3_family_grouped(all_results, out)
    plot_PD_ocsg_diverging(all_results, out)
    plot_PD7_response_length(all_results, out)
    plot_PD10_latency_per_model(all_results, out)

    # Block E
    plot_PE1_multi_radar(all_results, out)
    plot_PE2_umap_pca(all_results, df_flat, out)
    plot_PE3_ocsg_strip(all_results, out)
    plot_PE4_label_mix_alluvial(all_results, out)
    plot_PE5_bump_chart(all_results, out)

    # Block F — cross-model consistency (all new)
    plot_PF1_label_diversity_hist(all_results, out)
    plot_PF2_harm_variance_bubble(all_results, out)
    plot_PF3_pairwise_agreement_heatmap(all_results, out)
    plot_PF4_per_language_agreement(all_results, out)
    plot_PF5_model_agreement_strip(all_results, out)
    plot_PF6_disputed_label_tile(all_results, out)

    print(f"\n✅  All plots saved → {out.resolve()}")


if __name__ == "__main__":
    import argparse, sys
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--flat_csv",    default="results/enriched_flat.csv")
    p.add_argument("--output_dir",  default="results/figures")
    a = p.parse_args()
    all_results = {}
    for f in sorted(Path(a.results_dir).glob("*.csv")):
        try: all_results[f.stem] = pd.read_csv(f, index_col=0)
        except Exception as e: print(f"  ⚠ {f.name}: {e}")
    if not Path(a.flat_csv).exists():
        print(f"enriched_flat.csv not found", file=sys.stderr); sys.exit(1)
    plot_all(all_results, pd.read_csv(a.flat_csv), output_dir=a.output_dir)
