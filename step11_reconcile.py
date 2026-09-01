"""
Step 11: 效应量复核 —— 参数模型 vs 非参数事实
==============================================
参数(Poisson FE)估计隐含的"冬/夏病死率之比"偏大, 与简单配对(~1.09)不一致。
原因可能是: Poisson 按暴露量加权, 结果由加州/德州/佛州等大州主导;
而配对是"每州等权"。两者回答的问题不同, 都该报告。
这里用非参数方式把两种口径都算出来, 并做相互印证。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step10_refit import prep_tp, V2, spec
from step4_regression import poisson_fe


def nonparam(g):
    """剔除全局月效应后, 比较各地区寒季月 vs 暖季月的相对病死率(每地区等权)"""
    d = g.copy()
    d["log_cfr"] = np.log(d["deaths"] / d["confirmed"])
    # 全局(按半球)月效应
    d["month_fe"] = d.groupby(["hemi", "cal_month"])["log_cfr"].transform("mean")
    d["rel"] = d["log_cfr"] - d["month_fe"]
    rows = []
    for r, sub in d.groupby("region"):
        c, w = sub[sub.season == "cold"]["rel"], sub[sub.season == "warm"]["rel"]
        if len(c) < 3 or len(w) < 3:
            continue
        rows.append(dict(region=r, lat=sub["lat"].iloc[0], n_cold=len(c), n_warm=len(w),
                         dlog=c.mean() - w.mean(),
                         ratio=float(np.exp(c.mean() - w.mean()))))
    return pd.DataFrame(rows)


def weighted_nonparam(g):
    """按确诊量加权的版本: 合并所有地区后算总寒季/暖季归因死亡与确诊"""
    gg = g[g.season.isin(["cold", "warm"])]
    # 用各月全局相对因子做调整(等价于 month FE 的乘法版)
    tot = gg.groupby("cal_month").apply(lambda s: s.deaths.sum() / s.confirmed.sum(),
                                        include_groups=False).rename("month_cfr")
    gg = gg.merge(tot, on="cal_month")
    gg["expected"] = gg["confirmed"] * gg["month_cfr"]
    agg = gg.groupby("season").agg(obs=("deaths", "sum"), exp=("expected", "sum"),
                                   cases=("confirmed", "sum"))
    agg["oe"] = agg["obs"] / agg["exp"]
    return agg


if __name__ == "__main__":
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        for label, df_, dt_ in (("全期 2020-03~2022-12", "2020-03-01", "2022-12-31"),
                                ("剔除首轮 2020-09~", "2020-09-01", "2022-12-31")):
            g = prep_tp(scope, minc, 21, df_, dt_, V2)
            np_ = nonparam(g)
            r = np_["ratio"].to_numpy()
            w = stats.wilcoxon(np_["dlog"])
            par = spec(g)
            print(f"\n{'='*80}\n{tag} | {label}")
            print(f"  非参数(每地区等权): 中位寒/暖比 = {np.median(r):.3f}  "
                  f"几何均值 = {np.exp(np_.dlog.mean()):.3f}  "
                  f"寒季更差占比 = {100*(r>1).mean():.1f}%  Wilcoxon p = {w.pvalue:.4g}")
            print(f"  非参数(按确诊加权):")
            print(weighted_nonparam(g).to_string(float_format=lambda v: f"{v:.4f}"))
            print(f"  参数(Poisson FE)  : beta={par['beta']:+.4f} p={par['p']:.4g} "
                  f"每降10C: {par['pct_per_10degC']:+.1f}%")
            np_.to_csv(f"nonparam_{scope}_{'all' if '全期' in label else 'post'}.csv",
                       index=False, encoding="utf-8-sig")
