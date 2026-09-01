"""
Step 12: 拆解估计量 —— 参数模型的负系数究竟来自哪个维度的变异?
================================================================
双向 FE 的 beta 混合了两个来源:
  (i) 横截面维度: 同一日历月里, 更冷的州是否比更暖的州更糟
  (ii) 时序维度  : 同一个州, 它的冷月是否比它的暖月更糟
分别估计, 看负系数到底来自哪一边。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step10_refit import prep_tp, V2


def ols_fe(df, ycol, xcol, wcol, fe1, fe2, cluster):
    """加权最小二乘 + 双向去均值 + 聚类稳健 SE"""
    d = df.dropna(subset=[ycol, xcol, wcol]).copy()
    y = d[ycol].to_numpy(float)
    x = d[xcol].to_numpy(float)
    w = d[wcol].to_numpy(float)
    c1 = pd.factorize(d[fe1])[0]
    c2 = pd.factorize(d[fe2])[0]
    cl = pd.factorize(d[cluster])[0]
    n1, n2 = c1.max() + 1, c2.max() + 1

    def dm(v):
        for _ in range(300):
            p = v.copy()
            for c, n in ((c1, n1), (c2, n2)):
                s = np.bincount(c, weights=w * v, minlength=n)
                ws = np.bincount(c, weights=w, minlength=n)
                ws[ws == 0] = 1
                v = v - (s / ws)[c]
            if np.max(np.abs(v - p)) < 1e-12:
                break
        return v

    yd, xd = dm(y), dm(x)
    sw = np.sqrt(w)
    Y, X = yd * sw, xd * sw
    b = float((X @ Y) / (X @ X))
    res = Y - b * X
    XtX = float(X @ X)
    meat = sum(float((X[cl == g] @ res[cl == g]) ** 2) for g in np.unique(cl))
    G = len(np.unique(cl))
    se = np.sqrt((G / (G - 1)) * meat) / XtX
    return b, se, 2 * stats.norm.sf(abs(b / se)), len(d)


if __name__ == "__main__":
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        g = prep_tp(scope, minc, 21, "2020-03-01", "2022-12-31", V2)
        g["log_cfr"] = np.log(g["deaths"] / g["confirmed"])

        print(f"\n{'='*80}\n{tag}  (n={len(g)})")

        # --- (A) 双向 FE: OLS on log CFR (与 Poisson 对照)
        b, se, p, n = ols_fe(g, "log_cfr", "temp_c", "confirmed", "region", "hemi_month", "region")
        print(f"  [A] 双向FE OLS(log CFR,按确诊加权): beta={b:+.5f} se={se:.5f} p={p:.4g} "
              f"| 每降10C: {100*(np.exp(10*b)-1):+.1f}%")

        # 地区/月份的均值必须在**全面板**上算, 不能在分组内部算(否则退化成 0)
        g["t_rm"] = g.groupby("region")["temp_c"].transform("mean")
        g["y_rm"] = g.groupby("region")["log_cfr"].transform("mean")
        g["t_mm"] = g.groupby("hemi_month")["temp_c"].transform("mean")
        g["y_mm"] = g.groupby("hemi_month")["log_cfr"].transform("mean")

        # --- (B) 仅横截面维度: 每个日历月内, 跨地区做截面回归, 再按月平均
        slopes = []
        for m, sub in g.groupby("hemi_month"):
            if sub["region"].nunique() < 10:
                continue
            X = (sub["temp_c"] - sub["t_rm"]).to_numpy()
            Y = (sub["log_cfr"] - sub["y_rm"]).to_numpy()
            W = sub["confirmed"].to_numpy()
            msk = np.isfinite(X) & np.isfinite(Y) & (W > 0)
            if msk.sum() < 10 or X[msk].std() < 1e-9:
                continue
            Xc, Yc = X[msk] * np.sqrt(W[msk]), Y[msk] * np.sqrt(W[msk])
            slopes.append(float((Xc @ Yc) / (Xc @ Xc)))
        sl = np.array(slopes)
        t = stats.ttest_1samp(sl, 0)
        print(f"  [B] 横截面维度(同月内跨地区, {len(sl)}个月): 平均斜率={sl.mean():+.5f} "
              f"中位={np.median(sl):+.5f} t={t.statistic:+.2f} p={t.pvalue:.4g} "
              f"| 负斜率月份占比={100*(sl<0).mean():.0f}%")

        # --- (C) 仅时序维度: 每个地区内, 去掉全局月效应后对气温做时序回归
        ts = []
        for r, sub in g.groupby("region"):
            if len(sub) < 12:
                continue
            X = (sub["temp_c"] - sub["t_mm"]).to_numpy()
            Y = (sub["log_cfr"] - sub["y_mm"]).to_numpy()
            W = sub["confirmed"].to_numpy()
            msk = np.isfinite(X) & np.isfinite(Y) & (W > 0)
            if msk.sum() < 12 or X[msk].std() < 1e-9:
                continue
            Xc, Yc = X[msk] * np.sqrt(W[msk]), Y[msk] * np.sqrt(W[msk])
            ts.append(float((Xc @ Yc) / (Xc @ Xc)))
        ts = np.array(ts)
        t2 = stats.ttest_1samp(ts, 0)
        print(f"  [C] 时序维度(地区内跨月, {len(ts)}地区): 平均斜率={ts.mean():+.5f} "
              f"中位={np.median(ts):+.5f} t={t2.statistic:+.2f} p={t2.pvalue:.4g} "
              f"| 负斜率地区占比={100*(ts<0).mean():.0f}%")

        # --- (D) 横截面: 最冷月 vs 最暖月, 逐月看冷州-暖州的病死率差
        g["t_dev"] = g["temp_c"] - g.groupby("region")["temp_c"].transform("mean")
        g["cfr_dev"] = g["log_cfr"] - g.groupby("region")["log_cfr"].transform("mean")
        win = g[(g["cal_month"].str[5:7].isin(["01", "02", "07", "08"]))]
        for mm, nm in ((["01", "02"], "隆冬 1-2月"), (["07", "08"], "盛夏 7-8月")):
            s = win[win["cal_month"].str[5:7].isin(mm)]
            corr = stats.pearsonr(s["t_dev"], s["cfr_dev"])
            print(f"  [D] {nm}: 气温偏离 vs 病死率偏离 相关系数 r={corr.statistic:+.3f} "
                  f"p={corr.pvalue:.4g} (n={len(s)})")
