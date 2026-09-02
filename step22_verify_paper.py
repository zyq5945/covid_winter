# -*- coding: utf-8 -*-
"""核对论文关键数字 vs 现行 CSV（在 covid_test + py3.13 下运行）"""
import numpy as np, pandas as pd
from scipy import stats

print("=" * 70)
print("A. 4.1/4.2 汇总表 (final_penalty_table.csv)")
print("=" * 70)
t = pd.read_csv("final_penalty_table.csv")
for _, r in t.iterrows():
    print(f"{r['scope']:<16} {r['winter']:<12} {r['metric']:<10} "
          f"eff={r['eff']:>8.2f}%  share={r['share']:>6.2f}%  n={r['n']}  p={r['p']:.2e}")

print()
print("=" * 70)
print("B. 4.3 相关系数 (美国=每百万人口径, 北半球=绝对值口径)")
print("=" * 70)
us = pd.read_csv("winter_penalty_us_lag21.csv")
nh = pd.read_csv("hemi_global.csv").query("hemi == '北半球'")
for w in ("W1 2020冬-2021春", "W2 2021冬-2022春"):
    s = us[us.winter == w].dropna(subset=["penalty_deaths_pm", "temp_w"])
    r, p = stats.pearsonr(s["temp_w"], s["penalty_deaths_pm"])
    sr, sp = stats.spearmanr(s["temp_w"], s["penalty_deaths_pm"])
    print(f"美国 {w}: n={len(s)}  pearson={r:+.3f} (p={p:.4g})  spearman={sr:+.3f} (p={sp:.4g})")
s = us.dropna(subset=["penalty_deaths_pm", "temp_w"])
r, p = stats.pearsonr(s["temp_w"], s["penalty_deaths_pm"])
print(f"美国 合并: n={len(s)}  pearson={r:+.3f} (p={p:.4g})")
for w in ("W1 2020-21冬", "W2 2021-22冬"):
    s = nh[nh.winter == w].dropna(subset=["penalty_deaths", "temp_w"])
    r, p = stats.pearsonr(s["temp_w"], s["penalty_deaths"])
    sr, sp = stats.spearmanr(s["temp_w"], s["penalty_deaths"])
    print(f"北半球 {w}: n={len(s)}  pearson={r:+.3f} (p={p:.4g})  spearman={sr:+.3f} (p={sp:.4g})")
s = nh.dropna(subset=["penalty_deaths", "temp_w"])
r, p = stats.pearsonr(s["temp_w"], s["penalty_deaths"])
print(f"北半球 合并: n={len(s)}  pearson={r:+.3f} (p={p:.4g})")

print()
print("=" * 70)
print("B2. 4.3 病死率剂量-反应 (hemi_global.csv penalty_cfr)")
print("=" * 70)
h = pd.read_csv("hemi_global.csv")
for hemi in ("北半球", "南半球"):
    for w, s in h[h.hemi == hemi].groupby("winter"):
        d = s.dropna(subset=["penalty_cfr", "temp_w"])
        if len(d) < 6: continue
        r, p = stats.pearsonr(d["temp_w"], d["penalty_cfr"])
        print(f"{hemi} {w}: n={len(d)}  pearson={r:+.2f} (p={p:.4g})")
for hemi in ("北半球", "南半球"):
    d = h[h.hemi == hemi].dropna(subset=["penalty_cfr", "temp_w"])
    if len(d) < 6: continue
    r, p = stats.pearsonr(d["temp_w"], d["penalty_cfr"])
    print(f"{hemi} 合并: n={len(d)}  pearson={r:+.2f} (p={p:.4g})")

print()
print("=" * 70)
print("C. 4.4 半球宏观 + 合并 178 观测")
print("=" * 70)
for hemi in ("北半球", "南半球"):
    sub = h[h.hemi == hemi]
    print(f"{hemi}: 温差中位={ (sub['temp_s']-sub['temp_w']).median():.1f} ℃, "
          f"冬均温 {(sub['temp_w'].min()):.1f}~{(sub['temp_w'].max()):.1f} ℃, n_win={sub['winter'].nunique()}")
    for w, s in sub.groupby("winter"):
        pre = s["penalty_deaths_pre"].dropna()
        post = s["penalty_deaths_post"].dropna()
        both = s["penalty_deaths"].dropna()
        print(f"  {w}: 双侧 {100*(np.exp(both.mean())-1):+.1f}%  "
              f"前夏 {100*(np.exp(pre.mean())-1):+.1f}% ({100*(pre>0).mean():.0f}%)  "
              f"后夏 {100*(np.exp(post.mean())-1):+.1f}% ({100*(post>0).mean():.0f}%)")
v = h["penalty_deaths"].dropna()
wil = stats.wilcoxon(v)
print(f"合并 178 观测: n={len(v)}  中位 {100*(np.exp(v.median())-1):+.0f}%  "
      f"更糟占比 {100*(v>0).mean():.0f}%  Wilcoxon p={wil.pvalue:.2e}")

print()
print("=" * 70)
print("D. 4.1 月份分布 (fig4 口径: 寒季 11-3 / 暖季 5-9 / 过渡 4,10)")
print("=" * 70)
import cw_core as cw
from step10_refit import prep_tp
for scope, minc, tag in (("us", 100_000, "美国"), ("global", 200_000, "北半球国家")):
    g = prep_tp(scope, minc, 21, "2020-03-01", "2022-12-31", "daily_temperature_v2.csv")
    if scope == "global":
        g = g[g["lat"] >= 0]
    g["mon"] = g["date"].dt.month
    by = g.groupby("mon").agg(deaths=("deaths", "sum"), cases=("confirmed", "sum"))
    ds, cs = by.deaths, by.cases
    med_lat = g["lat"].median()
    def seas(m):
        s = cw.season_of(m, med_lat)
        return "寒" if s == "cold" else "暖" if s == "warm" else "过"
    grp = pd.Series([seas(m) for m in ds.index], index=ds.index)
    for name, ser in (("死亡", ds), ("确诊", cs)):
        share = (ser.groupby(grp).sum() / ser.sum() * 100).round(1)
        print(f"{tag} {name}: 寒季 {share.get('寒')}%  暖季 {share.get('暖')}%  过渡 {share.get('过')}%")
