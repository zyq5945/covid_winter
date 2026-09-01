"""
Step 6: 形状诊断 + 稳健性 —— 线性 beta 是否掩盖了阈值效应? 结果是否由个别地区驱动?
====================================================================================
6A 温度分箱: 用温度区间哑变量替代线性项, 看 CFR-温度关系是否为单调线性。
           若只有"低于某阈值"才恶化, 线性 beta 会低估; 若关系非单调, 线性 beta 无意义。
6B 留一地区法: 逐个剔除地区重估 Spec A, 检查结果是否由个别地区驱动。
6C 多 lag 与样本切分稳健性。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step4_regression import build_sample, poisson_fe

BINS = [-60, 0, 5, 10, 15, 20, 25, 60]
BLAB = ["<0C", "0-5C", "5-10C", "10-15C", "15-20C", "20-25C", ">25C"]


def binned(g, ref="20-25C"):
    gg = g.copy()
    gg["tbin"] = pd.cut(gg["temp_c"], BINS, labels=BLAB)
    gg = gg.dropna(subset=["tbin"])
    dummies = sorted([b for b in gg["tbin"].unique() if b != ref])
    for b in dummies:
        gg[f"tb_{b}"] = (gg["tbin"] == b).astype(float)
    if not dummies:
        return None, None
    tab, dg = poisson_fe(gg, y="deaths", offset_cols=["confirmed"],
                         x_cols=[f"tb_{b}" for b in dummies],
                         fe_cols=["region", "cal_month"], cluster="region")
    tab["bin"] = [t.replace("tb_", "") for t in tab["term"]]
    tab["pct_vs_ref"] = 100 * (np.exp(tab["beta"]) - 1)
    refrow = pd.DataFrame([{"bin": ref, "beta": 0.0, "se": 0.0, "z": np.nan, "p": np.nan,
                            "pct_vs_ref": 0.0, "term": "ref"}])
    out = pd.concat([tab[["bin", "beta", "se", "z", "p", "pct_vs_ref"]], refrow], ignore_index=True)
    order = [b for b in BLAB if b in set(out["bin"])]
    out["_o"] = out["bin"].map({b: i for i, b in enumerate(order)})
    return out.sort_values("_o").drop(columns="_o"), dg


def loo(g, xcol="temp_c"):
    rows = []
    for r in sorted(g["region"].unique()):
        sub = g[g["region"] != r]
        if sub["region"].nunique() < 10:
            continue
        tab, _ = poisson_fe(sub, y="deaths", offset_cols=["confirmed"], x_cols=[xcol],
                            fe_cols=["region", "cal_month"], cluster="region")
        rows.append(dict(dropped=r, beta=tab.iloc[0]["beta"], se=tab.iloc[0]["se"], p=tab.iloc[0]["p"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        g = build_sample(scope, minc, 21, "2020-03-01", "2022-12-31")
        print(f"\n{'='*80}\n{tag}  n={len(g)}  地区={g.region.nunique()}")

        print("\n[6A] 温度分箱 (参照 20-25C), Spec A: 地区FE + 日历月FE")
        tb, dg = binned(g)
        print(tb.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        tb.to_csv(f"tbins_{scope}.csv", index=False, encoding="utf-8-sig")

        print("\n[6B] 留一地区法 (Spec A 线性项)")
        lr = loo(g)
        base = poisson_fe(g, y="deaths", offset_cols=["confirmed"], x_cols=["temp_c"],
                          fe_cols=["region", "cal_month"], cluster="region")[0].iloc[0]
        print(f"  全样本 beta={base.beta:.4f} (p={base.p:.3g})")
        print(f"  留一后 beta 范围: [{lr.beta.min():.4f}, {lr.beta.max():.4f}]  "
              f"最大 p={lr.p.max():.3g}  显著(<0.05)占比={100*(lr.p<0.05).mean():.0f}%")
        print("  影响最大的 5 个地区:")
        lr["shift"] = (lr.beta - base.beta).abs()
        print(lr.nlargest(5, "shift")[["dropped", "beta", "p", "shift"]].to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
        lr.to_csv(f"loo_{scope}.csv", index=False, encoding="utf-8-sig")

        print("\n[6C] lag 稳健性 (Spec A)")
        rows = []
        for lag in (14, 21, 28):
            gl = build_sample(scope, minc, lag, "2020-03-01", "2022-12-31")
            tab, _ = poisson_fe(gl, y="deaths", offset_cols=["confirmed"], x_cols=["temp_c"],
                                fe_cols=["region", "cal_month"], cluster="region")
            rows.append(dict(lag=lag, n=len(gl), beta=tab.iloc[0]["beta"], se=tab.iloc[0]["se"],
                             p=tab.iloc[0]["p"], pct_per_10degC=tab.iloc[0]["pct_effect_per_10degC"]))
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
