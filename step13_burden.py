"""
Step 13: 换一个不受"检测口径"污染的结局 —— 死亡负担 与 传播强度
=====================================================================
CFR 的分母(确诊)受检测强度影响极大, 而检测强度本身有季节性(冬天检测需求更高),
这让 CFR 这个指标在季节性问题上不干净。改用两个更硬的结局:
  传播: E[确诊_{r,t}]   = exp(alpha_r + gamma_t + beta_T * T)      -> 冬天是否传播更强
  负担: E[死亡_{r,t}]   = exp(alpha_r + gamma_t + beta_T * T)      -> 冬天是否死更多人
  严重度差: beta_CFR ≈ beta_死亡 - beta_确诊
若 beta_传播 显著为负而 beta_负担 不显著 -> 冬天只是"感染的人多", 不是"感染者更容易死"。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step10_refit import prep_tp, V2
from step4_regression import poisson_fe
from step12_decompose import ols_fe


def fit_pois(g, ycol, offsets, fe=("region", "hemi_month")):
    tab, dg = poisson_fe(g, y=ycol, offset_cols=list(offsets), x_cols=["temp_c"],
                         fe_cols=list(fe), cluster="region")
    r = tab.iloc[0]
    return dict(beta=r.beta, se=r.se, p=r.p, pct_per_10degC=r.pct_effect_per_10degC, n=dg["n"])


if __name__ == "__main__":
    rows = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        for label, df_, dt_ in (("全期 2020-03~2022-12", "2020-03-01", "2022-12-31"),
                                ("剔除首轮 2020-09~", "2020-09-01", "2022-12-31")):
            g = prep_tp(scope, minc, 21, df_, dt_, V2)
            print(f"\n{'='*78}\n{tag} | {label} | n={len(g)}")

            rc = fit_pois(g, "confirmed", [])            # 传播
            rd = fit_pois(g, "deaths", [])               # 负担(相对)
            print(f"  传播 确诊数   beta={rc['beta']:+.4f} se={rc['se']:.4f} p={rc['p']:.4g} "
                  f"| 每降10C: {rc['pct_per_10degC']:+.1f}%")
            print(f"  负担 死亡数   beta={rd['beta']:+.4f} se={rd['se']:.4f} p={rd['p']:.4g} "
                  f"| 每降10C: {rd['pct_per_10degC']:+.1f}%")
            diff = rd["beta"] - rc["beta"]
            print(f"  隐含严重度差 (负担-传播) = {diff:+.4f} -> 每降10C: "
                  f"{100*(np.exp(10*diff)-1):+.1f}%")
            rows.append(dict(scope=tag, sample=label, outcome="传播:确诊", **rc))
            rows.append(dict(scope=tag, sample=label, outcome="负担:死亡", **rd))

            if scope == "us":     # 有人口 -> 人均死亡
                g2 = g.dropna(subset=["pop"]).copy()
                rp = fit_pois(g2, "deaths", ["pop"])
                b, se, p, n = ols_fe(g2.assign(ld=np.log(g2.deaths / g2.pop * 1e6)),
                                     "ld", "temp_c", "pop", "region", "hemi_month", "region")
                print(f"  人均死亡/百万  Poisson beta={rp['beta']:+.4f} se={rp['se']:.4f} "
                      f"p={rp['p']:.4g} | 每降10C: {rp['pct_per_10degC']:+.1f}%")
                print(f"  人均死亡/百万  OLS(log) beta={b:+.5f} se={se:.5f} p={p:.4g} "
                      f"| 每降10C: {100*(np.exp(10*b)-1):+.1f}%")
                rows.append(dict(scope=tag, sample=label, outcome="负担:人均死亡", **rp))

    out = pd.DataFrame(rows)
    out.to_csv("burden_transmission.csv", index=False, encoding="utf-8-sig")
    print("\n写出 burden_transmission.csv")
