"""Step 15: 生成最终图表"""
from __future__ import annotations
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

COLD = "#2b6cb0"      # 冬-蓝
WARM = "#dd6b20"      # 夏-橙
GREY = "#4a5568"

from step10_refit import prep_tp, V2


def pct(x, pos=None):
    return f"{x*100:.0f}%"


# ---------------------------------------------------------------- 数据
def monthly_us():
    g = prep_tp("us", 100_000, 21, "2020-03-01", "2022-12-31", V2)
    agg = g.groupby("date").agg(cases=("confirmed", "sum"), deaths=("deaths", "sum"),
                                pop=("pop", "sum"), temp=("temp_c", "mean")).reset_index()
    agg["cfr"] = agg.deaths / agg.cases
    agg["dpm"] = 1e6 * agg["deaths"] / agg["pop"]
    return agg


def agg_series(scope, minc):
    g = prep_tp(scope, minc, 21, "2020-03-01", "2022-12-31", V2)
    agg = g.groupby("date").agg(cases=("confirmed", "sum"), deaths=("deaths", "sum"),
                                temp=("temp_c", "mean")).reset_index()
    agg["cfr"] = agg.deaths / agg.cases
    return agg


# ---------------------------------------------------------------- 图1
def fig1():
    a = monthly_us()
    fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
    m = a["date"]
    ax[0].bar(m, a["cases"] / 1e6, color="#90cdf4", width=22)
    ax[0].set_ylabel("月新增确诊 (百万)")
    ax[0].set_title("美国合计：每月新增确诊 / 归因死亡 / 滞后对齐病死率，与月均气温对照",
                    fontsize=13, pad=10)
    ax[0].grid(axis="y", alpha=.3)

    ax[1].bar(m, a["deaths"] / 1e3, color=COLD, width=22)
    ax[1].set_ylabel("月归因死亡 (千)")
    ax[1].grid(axis="y", alpha=.3)

    ax[2].plot(m, a["cfr"], color="#c53030", lw=2, label="滞后对齐病死率 CFR")
    ax[2].set_ylabel("CFR")
    ax[2].grid(axis="y", alpha=.3)
    ax2 = ax[2].twinx()
    ax2.plot(m, a["temp"], color=WARM, lw=1.6, ls="--", alpha=.85, label="月均气温")
    ax2.set_ylabel("气温 (℃)", color=WARM)
    for i, (s, e, lb) in enumerate([("2020-11-01", "2021-03-31", "冬"),
                                    ("2021-11-01", "2022-03-31", "冬")]):
        for axx in ax:
            axx.axvspan(pd.Timestamp(s), pd.Timestamp(e), color="#bee3f8", alpha=.45, zorder=0)
    h1, l1 = ax[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax[2].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=9)
    ax[2].yaxis.set_major_formatter(FuncFormatter(pct))
    fig.savefig(f"{OUT}/fig1_us_monthly.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图2
def fig2():
    """决定性检验。季节按所在半球定义: 南半球国家的冬夏月份相反,
    混在一个池子里算会互相抵消, 所以全球样本必须按半球拆开。"""
    ORDER = ["新增确诊 (传播)", "死亡总数 (负担)", "病死率 (严重度)"]
    rows = []

    def add(tag, lab, col, nm, v):
        if col not in v.columns:
            return
        x = v[col].dropna().to_numpy()
        if len(x) < 6:
            return
        rows.append(dict(scope=tag, winter=lab, metric=nm,
                         eff=100 * (np.exp(x.mean()) - 1),
                         share=100 * (x > 0).mean(),
                         n=len(x), p=stats.wilcoxon(x).pvalue))

    # --- 美国 (全部位于北半球, 季节 = 11-3 月) ---
    d = pd.read_csv("winter_penalty_us_lag21.csv")
    for wname, s in d.groupby("winter"):
        lab = "2020-21 冬" if wname.startswith("W1") else "2021-22 冬"
        for col, nm in (("penalty_cfr", "病死率 (严重度)"),
                        ("penalty_deaths_pm", "死亡总数 (负担)"),
                        ("penalty_cases", "新增确诊 (传播)")):
            add("美国 52 个地区", lab, col, nm, s)

    # --- 全球, 按半球分别算 ---
    g = pd.read_csv("hemi_global.csv")
    for hemi in ("北半球", "南半球"):
        h = g[g.hemi == hemi]
        tag = f"全球{hemi} {h.region.nunique()} 国"
        for wname, s in h.groupby("winter"):
            if hemi == "南半球":
                lab = "2021 冬"          # 南半球只有一个可用窗口
            else:
                lab = "2020-21 冬" if wname.startswith("W1") else "2021-22 冬"
            for col, nm in (("penalty_cfr", "病死率 (严重度)"),
                            ("penalty_deaths", "死亡总数 (负担)"),
                            ("penalty_cases", "新增确诊 (传播)")):
                add(tag, lab, col, nm, s)

    r = pd.DataFrame(rows)
    r.to_csv("final_penalty_table.csv", index=False, encoding="utf-8-sig")

    scopes = [s for s in ("美国 52 个地区",
                          *[f"全球{h} {g[g.hemi==h].region.nunique()} 国" for h in ("北半球", "南半球")])]
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.4))
    for ax, sc in zip(axes, scopes):
        sub = r[r.scope == sc]
        piv = sub.pivot_table(index="metric", columns="winter", values="eff").reindex(ORDER)
        x = np.arange(len(ORDER))
        w = 0.36
        sh_p = sub.pivot_table(index="metric", columns="winter", values="share").reindex(ORDER)
        pv = sub.pivot_table(index="metric", columns="winter", values="p").reindex(ORDER)
        for i, col in enumerate(piv.columns):
            vals = piv[col].to_numpy()
            cols = [COLD if v > 0 else "#c53030" for v in vals]
            b = ax.bar(x + (i - .5) * w, vals, w, label=col, color=cols, alpha=.9)
            for rect, v, sh, p in zip(b, vals, sh_p[col], pv[col]):
                star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
                ax.text(rect.get_x() + rect.get_width() / 2,
                        v + (12 if v > 0 else -22), f"{v:+.0f}%\n{sh:.0f}%地区 {star}",
                        ha="center", va="bottom" if v > 0 else "top", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ORDER, fontsize=9.5)
        ax.axhline(0, color="k", lw=.8)
        ax.set_ylabel("冬季相对前后两个夏天的变化 (%)")
        ax.set_title(sc, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=.3)
        ax.set_ylim(-70, 300)
    fig.suptitle("决定性检验：每个冬天 vs 它前后两个夏天的平均（季节按所在半球定义）",
                 fontsize=13, y=1.02)
    fig.savefig(f"{OUT}/fig2_penalty.png")
    plt.close(fig)
    return r


# ---------------------------------------------------------------- 图3
def fig3():
    """剂量-反应散点。右侧只画北半球国家: 南半球国家的"冬季"是 5-9 月,
    与北半球混在一张图里会互相抵消 (见 step19_hemisphere.py)。"""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    panels = (
        ("美国 52 个地区", pd.read_csv("winter_penalty_us_lag21.csv"),
         "penalty_deaths_pm", ("W1 2020冬-2021春", "W2 2021冬-2022春")),
        ("全球北半球国家", pd.read_csv("hemi_global.csv").query("hemi == '北半球'"),
         "penalty_deaths", ("W1 2020-21冬", "W2 2021-22冬")),
    )
    for ax, (tag, d, ycol, wnames) in zip(axes, panels):
        for wname, mk in zip(wnames, ("o", "s")):
            lab = wname.split(" ", 1)[-1]
            s = d[d.winter == wname].dropna(subset=[ycol, "temp_w"])
            ax.scatter(s["temp_w"], 100 * (np.exp(s[ycol]) - 1), s=42, alpha=.75,
                       marker=mk, label=lab, edgecolor="white", linewidth=.5)
        ax.axhline(0, color="k", lw=.9)
        ax.set_xlabel("该地区冬季平均气温 (℃)")
        ax.set_ylabel("冬季相对夏季的死亡变化 (%)")
        ax.set_title(f"{tag}：冬季严寒程度 vs 冬季死亡惩罚", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=.3)
        # 趋势线
        s = d.dropna(subset=[ycol, "temp_w"])
        if len(s) > 10:
            sl, ic = np.polyfit(s["temp_w"], 100 * (np.exp(s[ycol]) - 1), 1)
            xs = np.linspace(s["temp_w"].min(), s["temp_w"].max(), 20)
            r, p = stats.pearsonr(s["temp_w"], s[ycol])
            ax.plot(xs, sl * xs + ic, color="#c53030", lw=2, ls="--",
                    label=f"趋势 r={r:+.2f} p={p:.3g}")
            ax.legend(fontsize=9)
    fig.suptitle("越冷的地区，冬季死亡惩罚越大吗？（剂量-反应检验）", fontsize=13, y=1.03)
    fig.savefig(f"{OUT}/fig3_scatter.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图4
def fig4():
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    for j, (scope, minc, tag) in enumerate((("us", 100_000, "美国 50 州+DC"),
                                            ("global", 200_000, "全球北半球国家"))):
        g = prep_tp(scope, minc, 21, "2020-03-01", "2022-12-31", V2)
        # 季节按所在半球定义: 南半球国家的冬夏月份相反, 混在一张图里会互相抵消
        if scope == "global":
            g = g[g["lat"] >= 0]
        g["mon"] = g["date"].dt.month
        by = g.groupby("mon").agg(cases=("confirmed", "sum"), deaths=("deaths", "sum"))
        by["dshare"] = by.deaths / by.deaths.sum()
        by["cshare"] = by.cases / by.cases.sum()
        ax = axes[0][j]
        cols = [COLD if cw_.season_of(m, g["lat"].median()) == "cold"
                else WARM if cw_.season_of(m, g["lat"].median()) == "warm" else "#a0aec0"
                for m in by.index]
        ax.bar(by.index, by.dshare, color=cols)
        ax.set_title(f"{tag}：各月死亡占全期比重", fontsize=12)
        ax.set_xlabel("月份")
        ax.set_ylabel("占全期死亡比重")
        ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.set_xticks(range(1, 13))
        ax.grid(axis="y", alpha=.3)
        cold_share = by.loc[[11, 12, 1, 2, 3], "dshare"].sum()
        warm_share = by.loc[[5, 6, 7, 8, 9], "dshare"].sum()
        ax.text(.5, .93, f"寒季(11-3月) {cold_share*100:.1f}%  |  暖季(5-9月) {warm_share*100:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", fc="#fff5e6", ec=WARM))

        ax = axes[1][j]
        ax.bar(by.index, by.cshare, color=cols)
        ax.set_title(f"{tag}：各月确诊占全期比重", fontsize=12)
        ax.set_xlabel("月份")
        ax.set_ylabel("占全期确诊比重")
        ax.yaxis.set_major_formatter(FuncFormatter(pct))
        ax.set_xticks(range(1, 13))
        ax.grid(axis="y", alpha=.3)
        cs = by.loc[[11, 12, 1, 2, 3], "cshare"].sum()
        ws = by.loc[[5, 6, 7, 8, 9], "cshare"].sum()
        ax.text(.5, .93, f"寒季(11-3月) {cs*100:.1f}%  |  暖季(5-9月) {ws*100:.1f}%",
                transform=ax.transAxes, ha="center", fontsize=10,
                bbox=dict(boxstyle="round", fc="#ebf8ff", ec=COLD))
    fig.suptitle("死亡与确诊在一年中的分布（蓝=寒季月，橙=暖季月，灰=过渡月；按中位数纬度判半球）",
                 fontsize=12.5, y=.99)
    fig.tight_layout(rect=[0, 0, 1, .96])
    fig.savefig(f"{OUT}/fig4_monthshare.png")
    plt.close(fig)


# ---------------------------------------------------------------- 图5 数据质量
def fig5():
    qc = pd.concat([pd.read_csv("qc_global.csv").assign(scope="全球国家"),
                    pd.read_csv("qc_us.csv").assign(scope="美国州")])
    qc = qc[qc.rev_pct > 0]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, (sc, s) in zip(axes, qc.groupby("scope")):
        s = s.nlargest(12, "rev_pct").sort_values("rev_pct")
        ax.barh(s["region"], s["rev_pct"], color="#805ad5")
        ax.set_title(f"{sc}：数据回调（负增量）最严重的地区", fontsize=12)
        ax.set_xlabel("累计回调量占最终累计确诊的比重 (%)")
        ax.grid(axis="x", alpha=.3)
    fig.suptitle("数据质量：JHU 累计序列的向下修订（本分析已用回溯剥离法修复）",
                 fontsize=12.5, y=1.03)
    fig.savefig(f"{OUT}/fig5_dataqc.png")
    plt.close(fig)


if __name__ == "__main__":
    import cw_core as cw_
    fig1(); print("fig1 OK")
    r = fig2(); print("fig2 OK"); print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    fig3(); print("fig3 OK")
    fig4(); print("fig4 OK")
    fig5(); print("fig5 OK")
