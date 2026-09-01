"""
Step 5: 异质性与剂量-反应 —— 为什么跨国样本"看不到"寒冷效应
============================================================
假设: 加入地区FE+日历月FE 后, 识别变异来自"各地区相对自身年均值的季节性偏离"。
热带国家全年温差接近 0, 识别信息几乎为零, 却贡献了大量噪声 -> 稀释跨国估计。
若"寒冷有害"为真, 应看到 **季节温差幅度越大的地区, beta 越负** (剂量-反应)。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step4_regression import build_sample, poisson_fe, DEFAULT_LAG


def spec_a(g, extra_x=(), fe_extra=()):
    tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"],
                         x_cols=["temp_c"] + list(extra_x),
                         fe_cols=["region", "cal_month"] + list(fe_extra),
                         cluster="region")
    return tab, dg


def by_group(g, group_col, label):
    rows = []
    med = g.groupby("region")[group_col].median()
    g = g.copy()
    g["_grp_val"] = g["region"].map(med)
    try:
        g["_grp"] = pd.qcut(g["_grp_val"], 3, labels=["低", "中", "高"])
    except ValueError:
        g["_grp"] = "ALL"
    for grp, sub in g.groupby("_grp", observed=True):
        if sub["region"].nunique() < 8 or len(sub) < 150:
            continue
        tab, dg = spec_a(sub)
        r = tab.iloc[0]
        rows.append(dict(scope=label, group=f"{group_col}{grp}", n_regions=sub["region"].nunique(),
                         n_obs=len(sub), grp_value=float(sub["_grp_val"].median()),
                         beta=r.beta, se=r.se, z=r.z, p=r.p,
                         pct_per_10degC=r.pct_effect_per_10degC))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        g = build_sample(scope, minc, DEFAULT_LAG, "2020-03-01", "2022-12-31")
        print(f"\n{'='*80}\n{tag}: {len(g)} 地区-月, {g.region.nunique()} 地区")
        print(f"  季节温差幅度分布: {g.groupby('region')['amplitude'].median().describe()[['25%','50%','75%']].to_dict()}")

        r1 = by_group(g, "amplitude", tag)
        print("\n  [剂量-反应] 按地区季节温差幅度三分位 (Spec A: 地区FE + 日历月FE)")
        print(r1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        out.append(r1)

        # 非热带子集
        for thr, nm in ((8, "温差幅度>8C"), (12, "温差幅度>12C")):
            sub = g.groupby("region").filter(lambda x: x["amplitude"].median() >= thr)
            if sub["region"].nunique() >= 10:
                tab, dg = spec_a(sub)
                r = tab.iloc[0]
                print(f"\n  [{nm}] n_region={sub.region.nunique()}, n={len(sub)} | "
                      f"beta={r.beta:.4f} se={r.se:.4f} p={r.p:.4g} "
                      f"| 每降10C: {r.pct_effect_per_10degC:+.1f}%")
                out.append(pd.DataFrame([dict(scope=tag, group=nm, n_regions=sub.region.nunique(),
                                             n_obs=len(sub), grp_value=np.nan, beta=r.beta, se=r.se,
                                             z=r.z, p=r.p, pct_per_10degC=r.pct_effect_per_10degC)]))

    res = pd.concat(out, ignore_index=True)
    res.to_csv("heterogeneity.csv", index=False, encoding="utf-8-sig")
    print("\n写出 heterogeneity.csv")
