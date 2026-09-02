"""
Step 14: 决定性检验 —— 用"相邻两个夏天"给冬天做对照
======================================================
设计
  对每个地区, 把每个冬天夹在它前一个夏天和后一个夏天中间:
      penalty(W) = log(CFR_W) - 0.5 * [log(CFR_Sbefore) + log(CFR_Safter)]
  因为对照是"前后两个夏天的平均", 任何**平滑的时间趋势**(疫苗铺开、变异株更替、
  治疗进步、检测扩容)在夏天到冬天的插值上被一阶抵消。剩下的才是季节效应。

  冬季序列:  W1 = 2020-11~2021-03,  W2 = 2021-11~2022-03
  夏季序列:  S0 = 2020-05~2020-09,  S1 = 2021-05~2021-09,  S2 = 2022-05~2022-09
  W1 对照 (S0,S1);  W2 对照 (S1,S2)

判定
  penalty > 0 表示"这个地区的冬天比它前后两个夏天更致命"。
  若多数地区 penalty > 0 且符号检验显著 -> 支持假说。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step10_refit import prep_tp, V2

# (冬季标签, 起始月, 结束月(次年), 前一个夏天, 后一个夏天)
DESIGN = [
    ("W1 2020冬-2021春", "2020-11-01", "2021-03-31", ("2020-05-01", "2020-09-30"), ("2021-05-01", "2021-09-30")),
    ("W2 2021冬-2022春", "2021-11-01", "2022-03-31", ("2021-05-01", "2021-09-30"), ("2022-05-01", "2022-09-30")),
]


def _logratio(w, s1, s2):
    """
    log(w) - 0.5*[log(s1) + log(s2)]，带合法性检查。

    直接算会在三种情况下炸出 RuntimeWarning 或 NaN：
      - 任一项为 0        -> log(0) = -inf，两个 -inf 相减得到 NaN
      - 任一项为 inf      -> 人口为 0 时 deaths_pm = inf，inf - inf = NaN
      - 任一项为 NaN      -> 人口缺失时整列是 NaN
    这三种都不该静默产出假数字，统一返回 NaN，由下游 dropna 剔除。
    """
    vals = (w, s1, s2)
    if any(v is None or not np.isfinite(v) or v <= 0 for v in vals):
        return np.nan
    return float(np.log(w) - 0.5 * (np.log(s1) + np.log(s2)))


def season_agg(g, start, end, val_cols):
    m = (g["date"] >= pd.Timestamp(start)) & (g["date"] <= pd.Timestamp(end))
    s = g[m]
    if len(s) == 0:
        return None
    out = {"months": len(s)}
    for c in val_cols:
        out[c] = s[c].sum()
    out["temp_c"] = s["temp_c"].mean()
    return out


def run(scope, minc, tag, tp=V2, lag=21):
    g = prep_tp(scope, minc, lag, "2020-03-01", "2022-12-31", tp)
    g["log_cfr"] = np.log(g["deaths"] / g["confirmed"])
    have_pop = "pop" in g.columns and g["pop"].notna().any()
    if have_pop:
        g["deaths_pm"] = 1e6 * g["deaths"] / g["pop"]
        g["cases_pm"] = 1e6 * g["confirmed"] / g["pop"]

    cols = ["confirmed", "deaths"] + (["deaths_pm", "cases_pm"] if have_pop else [])
    rows = []
    for wname, ws, we, (s1a, s1b), (s2a, s2b) in DESIGN:
        for r, sub in g.groupby("region"):
            W = season_agg(sub, ws, we, cols)
            S1 = season_agg(sub, s1a, s1b, cols)
            S2 = season_agg(sub, s2a, s2b, cols)
            if not W or not S1 or not S2:
                continue
            if min(W["confirmed"], S1["confirmed"], S2["confirmed"]) < 2000:
                continue
            rec = dict(region=r, lat=sub["lat"].iloc[0], winter=wname,
                       cfr_w=W["deaths"] / W["confirmed"],
                       cfr_s1=S1["deaths"] / S1["confirmed"],
                       cfr_s2=S2["deaths"] / S2["confirmed"],
                       temp_w=W["temp_c"], temp_s=0.5 * (S1["temp_c"] + S2["temp_c"]),
                       cases_w=W["confirmed"], cases_s=0.5 * (S1["confirmed"] + S2["confirmed"]))
            rec["penalty_cfr"] = _logratio(rec["cfr_w"], rec["cfr_s1"], rec["cfr_s2"])
            rec["penalty_cases"] = _logratio(W["confirmed"], S1["confirmed"], S2["confirmed"])
            if have_pop:
                rec["penalty_deaths_pm"] = _logratio(W["deaths_pm"], S1["deaths_pm"], S2["deaths_pm"])
                rec["penalty_cases_pm"] = _logratio(W["cases_pm"], S1["cases_pm"], S2["cases_pm"])
            else:
                rec["penalty_deaths"] = _logratio(W["deaths"], S1["deaths"], S2["deaths"])
            rows.append(rec)
    return pd.DataFrame(rows)


def report(df, tag):
    print(f"\n{'='*80}\n{tag}")
    metrics = [c for c in df.columns if c.startswith("penalty_")]
    for wname, sub in df.groupby("winter"):
        print(f"\n  --- {wname}  (n={len(sub)} 地区) ---")
        print(f"    冬季均温 {sub.temp_w.mean():.1f}C vs 对照夏季均温 {sub.temp_s.mean():.1f}C "
              f"(温差 {sub.temp_s.mean()-sub.temp_w.mean():.1f}C)")
        for m in metrics:
            v = sub[m].dropna().to_numpy()
            if len(v) < 8:
                continue
            w = stats.wilcoxon(v)
            s = stats.binomtest(int((v > 0).sum()), len(v), 0.5)
            ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            print(f"    {m:22s} 冬季相对对照: {100*(np.exp(v.mean())-1):+7.1f}%  "
                  f"95%CI[{100*(np.exp(ci[0])-1):+6.1f}%,{100*(np.exp(ci[1])-1):+6.1f}%]  "
                  f"更差占比 {100*(v>0).mean():4.1f}%  Wilcoxon p={w.pvalue:.4f}  符号 p={s.pvalue:.4f}")


if __name__ == "__main__":
    for lag in (14, 21, 28):
        print(f"\n\n{'#'*80}\n# 滞后对齐 lag = {lag} 天\n{'#'*80}")
        for scope, minc, tag in (("us", 100_000, "美国 50 州+DC"), ("global", 200_000, "全球国家")):
            df = run(scope, minc, tag, V2, lag)
            report(df, tag)
            df.to_csv(f"winter_penalty_{scope}_lag{lag}.csv", index=False, encoding="utf-8-sig")
