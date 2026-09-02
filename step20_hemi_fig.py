"""Step 20: 半球对比图"""
from __future__ import annotations
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DengXian"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

OUT = "figures"
COLD, WARM, GREY, RED = "#2b6cb0", "#dd6b20", "#4a5568", "#c53030"

d = pd.read_csv("hemi_global.csv")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# ---- 左: 各半球/各窗口的死亡超额 + 单侧对照分解
ax = axes[0]
labels, vals, cols, errs = [], [], [], []
for hemi in ("北半球", "南半球"):
    h = d[d.hemi == hemi]
    for w, s in h.groupby("winter"):
        yr = w.split()[1]
        for col, lab in (("penalty_deaths", "双侧平均"),
                         ("penalty_deaths_pre", "仅对照前夏"),
                         ("penalty_deaths_post", "仅对照后夏")):
            v = s[col].dropna().to_numpy()
            if len(v) < 6:
                continue
            labels.append(f"{hemi[:2]}\n{yr}\n{lab}")
            vals.append(100 * (np.exp(v.mean()) - 1))
            errs.append(100 * (np.exp(v.mean() + stats.sem(v)) - np.exp(v.mean())))
            cols.append(COLD if hemi == "北半球" else WARM)
x = np.arange(len(labels))
b = ax.bar(x, vals, color=cols, alpha=.85, yerr=errs, capsize=3,
           error_kw=dict(elinewidth=1, ecolor=GREY))
for r, v in zip(b, vals):
    ax.text(r.get_x() + r.get_width() / 2, v + 14, f"{v:+.0f}%",
            ha="center", fontsize=8.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8.5)
ax.axhline(0, color="k", lw=.8)
ax.set_ylabel("冬季相对夏季的死亡变化 (%)")
ax.set_title("冬季死亡超额：双侧对照 vs 单侧对照\n（蓝=北半球，橙=南半球）", fontsize=11.5)
ax.grid(axis="y", alpha=.3)

# ---- 右: 宏观剂量-反应: 冬夏温差 vs 死亡超额
ax = axes[1]
pts = []
for hemi, h in d.groupby("hemi"):
    sw = (h.temp_s - h.temp_w).median()      # 半球级冬夏温差中位数 (北 14.3 / 南 4.7)
    n = h.region.nunique()                   # 半球国家总数 (北 93 / 南 15)
    if hemi == "北半球":
        # 宏观点只用 2020-21 冬这个最干净的窗口: 它的后一个夏天正是奥密克戎,
        # 而前一个夏天(2020 夏)疫情小, 用"仅前夏"做对照才不掺毒株更替 (见 4.4 节正文,
        # 北半球干净对照 = +208%)。把两个冬天合起来取平均会被 W2 的 +12.5% 稀释成 +83%。
        h = h[h.winter == "W1 2020-21冬"]
    v_all = h.penalty_deaths.dropna()
    v_pre = h.penalty_deaths_pre.dropna()
    pts.append((hemi, sw, 100 * (np.exp(v_all.mean()) - 1),
                100 * (np.exp(v_pre.mean()) - 1), n))
for hemi, sw, y1, y2, n in pts:
    c = COLD if hemi == "北半球" else WARM
    ax.scatter(sw, y1, s=260, color=c, marker="o", zorder=3,
               edgecolor="white", linewidth=1.5, label=f"{hemi}（双侧对照）")
    ax.scatter(sw, y2, s=260, color=c, marker="s", zorder=3, alpha=.55,
               edgecolor="white", linewidth=1.5, label=f"{hemi}（避开奥密克戎的干净对照）")
    # 标签放左下, 避开标题
    ax.annotate(f"{hemi} ({n} 国)", (sw, y1), textcoords="offset points",
                xytext=(-12, 0), ha="right", va="center", fontsize=9.5, color=c)
xs = [p[1] for p in pts]
ys = [p[3] for p in pts]
ax.plot(xs, ys, color=GREY, lw=1.6, ls="--", zorder=1)
ax.set_xlabel("冬夏温差中位数 (℃)")
ax.set_ylabel("冬季死亡超额 (%)")
ax.set_title("宏观剂量-反应: 冬夏温差越大, 冬季死亡惩罚越大", fontsize=11.5)
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=.3)
ax.set_xlim(0, 18)
ax.set_ylim(0, 260)

fig.suptitle("南半球 vs 北半球：两个相隔半年的冬天，给出同一个方向",
             fontsize=13.5, y=1.04)
fig.savefig(f"{OUT}/fig6_hemisphere.png")
plt.close(fig)
print("写出 figures/fig6_hemisphere.png")
for p in pts:
    print(f"  {p[0]}: 温差 {p[1]:.1f}℃  双侧 {p[2]:+.1f}%  干净对照 {p[3]:+.1f}%  n={p[4]}")
