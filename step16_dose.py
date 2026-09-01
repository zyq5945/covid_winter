"""Step 16: 剂量-反应回归 + 报告汇编"""
from __future__ import annotations
import os, json
import numpy as np, pandas as pd
from scipy import stats


def dose(d, temp_col, ycol):
    d = d.dropna(subset=[temp_col, ycol]).copy()
    x = d[temp_col].to_numpy()
    y = d[ycol].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r, p = stats.pearsonr(x, y)
    sp, sp_p = stats.spearmanr(x, y)
    sl, ic, *_ = stats.linregress(x, y)
    return dict(n=len(x), pearson_r=r, pearson_p=p, spearman_r=sp, spearman_p=sp_p,
                slope_pct_per_C=float(sl), intercept_pct=float(ic),
                median_penalty_pct=float(np.median(100 * (np.exp(y) - 1))))


if __name__ == "__main__":
    rows = []
    for scope in ("us", "global"):
        d = pd.read_csv(f"winter_penalty_{scope}_lag21.csv")
        ycol = "penalty_deaths_pm" if scope == "us" else "penalty_deaths"
        # 死亡惩罚 表达为 100*(exp(y)-1) 即百分比
        d["y_pct"] = 100 * (np.exp(d[ycol]) - 1)
        for w, sub in d.groupby("winter"):
            r = dose(sub.assign(_y=sub["penalty_deaths_pm" if scope == "us" else "penalty_deaths"]),
                     "temp_w", "_y")
            # 原始 (log 尺度) 回归
            r.update(scope=scope, winter=w, outcome="死亡惩罚")
            rows.append(r)
            # 包含大小国家, 用确诊量加权
            wgt = sub["cases_w"].to_numpy()
            x = sub["temp_w"].to_numpy()
            yy = sub["penalty_deaths_pm" if scope == "us" else "penalty_deaths"].to_numpy()
            ok = np.isfinite(x) & np.isfinite(yy) & (wgt > 0)
            if ok.sum() > 10:
                Xc = x[ok] * np.sqrt(wgt[ok])
                Yc = yy[ok] * np.sqrt(wgt[ok])
                b = float((Xc @ Yc) / (Xc @ Xc))
                r.update(weighted_slope_per_C=b)
            # 把结果保存
        d.to_csv(f"dose_{scope}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv("dose_response.csv", index=False, encoding="utf-8-sig")

    # 全球: 死亡惩罚 与 季节温差幅度
    dg = pd.read_csv("winter_penalty_global_lag21.csv")
    if "temp_w" in dg.columns:
        # 没有幅度, 暂时不展开
        pass

    # 也算 CFR 惩罚 与 冬季气温
    rows2 = []
    for scope in ("us", "global"):
        d = pd.read_csv(f"winter_penalty_{scope}_lag21.csv")
        for w, sub in d.groupby("winter"):
            r = dose(sub, "temp_w", "penalty_cfr")
            r.update(scope=scope, winter=w, outcome="CFR惩罚")
            rows2.append(r)
    pd.DataFrame(rows2).to_csv("dose_response_cfr.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(rows2).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
