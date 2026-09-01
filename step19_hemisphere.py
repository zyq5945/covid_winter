"""
Step 19: 按南北半球分别计算冬夏
================================
发现的问题: step14 的季节窗口是按日历月硬编码的(11-3 月 = 冬、5-9 月 = 夏),
这对南半球 17 个国家是**反的**——它们的 11-3 月是夏天。
这 17 国占全球样本死亡数的 22.1%, 是实质性污染, 必须分开重算。

修正后的设计
  北半球(NH): 冬 = 11/1 - 3/31, 夏 = 5/1 - 9/30
  南半球(SH): 冬 = 5/1 - 9/30,  夏 = 11/1 - 3/31

窗口可用性(数据 2020-01-22 ~ 2023-03-09)
  NH: 2 个冬天 (2020-21, 2021-22), 前后夏天均完整
  SH: 3 个冬天 (2020, 2021, 2022)
        - SH 2021 冬: 前后两个夏天(2020-21, 2021-22)均完整  <- 主结果
        - SH 2020 冬: 前一个夏天(2019-20)因数据起于 1/22 而残缺
        - SH 2022 冬: 后一个夏天(2022-23)因数据止于 3/9 而残缺

处理办法: 对"水平类"指标(确诊数、死亡数)按**月均**计算, 使长短不一的窗口可比;
         病死率本身就是比值, 无需归一化。每个冬天标注对照夏天的完整度。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step10_refit import prep_tp, V2

# (标签, 冬起, 冬止, (夏1起,夏1止), (夏2起,夏2止))
NH = [
    ("W1 2020-21冬", "2020-11-01", "2021-03-31", ("2020-05-01", "2020-09-30"), ("2021-05-01", "2021-09-30")),
    ("W2 2021-22冬", "2021-11-01", "2022-03-31", ("2021-05-01", "2021-09-30"), ("2022-05-01", "2022-09-30")),
]
SH = [
    ("W1 2020冬", "2020-05-01", "2020-09-30", ("2019-11-01", "2020-03-31"), ("2020-11-01", "2021-03-31")),
    ("W2 2021冬", "2021-05-01", "2021-09-30", ("2020-11-01", "2021-03-31"), ("2021-11-01", "2022-03-31")),
    ("W3 2022冬", "2022-05-01", "2022-09-30", ("2021-11-01", "2022-03-31"), ("2022-11-01", "2023-03-31")),
]

MIN_MONTHS = 3          # 每个窗口至少要有几个完整月
MIN_CASES = 2000        # 每个窗口累计确诊下限


def window_agg(sub, start, end, cols):
    m = (sub["date"] >= pd.Timestamp(start)) & (sub["date"] <= pd.Timestamp(end))
    s = sub[m]
    if len(s) == 0:
        return None
    out = {"months": len(s), "temp": s["temp_c"].mean()}
    for c in cols:
        out[c] = s[c].sum()
    return out


def run(scope, minc, tp=V2, lag=21):
    g = prep_tp(scope, minc, lag, "2020-03-01", "2022-12-31", tp)
    g["south"] = g["lat"] < 0
    have_pop = "pop" in g.columns and g["pop"].notna().any()
    cols = ["confirmed", "deaths"] + (["pop"] if have_pop else [])
    if have_pop:
        g["deaths_pm_rate"] = 1e6 * g["deaths"] / g["pop"]

    rows = []
    for hemi, design in (("北半球", NH), ("南半球", SH)):
        sub_all = g[g["south"] == (hemi == "南半球")]
        if sub_all["region"].nunique() == 0:
            continue
        for wname, ws, we, (a1, a2), (b1, b2) in design:
            for r, sub in sub_all.groupby("region"):
                W = window_agg(sub, ws, we, cols)
                S1 = window_agg(sub, a1, a2, cols)
                S2 = window_agg(sub, b1, b2, cols)
                if not W or not S1 or not S2:
                    continue
                if min(W["months"], S1["months"], S2["months"]) < MIN_MONTHS:
                    continue
                if min(W["confirmed"], S1["confirmed"], S2["confirmed"]) < MIN_CASES:
                    continue

                def rate(d, c):        # 月均
                    return max(d[c], 1e-9) / d["months"]

                rec = dict(hemi=hemi, region=r, lat=sub["lat"].iloc[0], winter=wname,
                           m_w=W["months"], m_s1=S1["months"], m_s2=S2["months"],
                           temp_w=W["temp"], temp_s=(S1["temp"] + S2["temp"]) / 2,
                           cases_w=W["confirmed"])
                # 水平类指标: 按月均比较
                for c, nm in (("confirmed", "cases"), ("deaths", "deaths")):
                    rec[f"penalty_{nm}"] = (np.log(rate(W, c))
                                            - 0.5 * (np.log(rate(S1, c)) + np.log(rate(S2, c))))
                    # 单侧对照分解: 单独用前一个 / 后一个夏天, 用于识别"对照被污染"的情形
                    rec[f"penalty_{nm}_pre"] = np.log(rate(W, c)) - np.log(rate(S1, c))
                    rec[f"penalty_{nm}_post"] = np.log(rate(W, c)) - np.log(rate(S2, c))
                # 病死率: 比值, 无需归一化
                cfr = lambda d: d["deaths"] / d["confirmed"]
                rec["penalty_cfr"] = np.log(cfr(W)) - 0.5 * (np.log(cfr(S1)) + np.log(cfr(S2)))
                rec["penalty_cfr_pre"] = np.log(cfr(W)) - np.log(cfr(S1))
                rec["penalty_cfr_post"] = np.log(cfr(W)) - np.log(cfr(S2))
                # 美国有 pop -> 人均死亡月均
                if have_pop and W.get("pop"):
                    pm = lambda d: 1e6 * d["deaths"] / d["pop"] / d["months"]
                    rec["penalty_deaths_pm"] = np.log(pm(W)) - 0.5 * (np.log(pm(S1)) + np.log(pm(S2)))
                rows.append(rec)
    return pd.DataFrame(rows)


def report(df, tag):
    print(f"\n{'='*84}\n{tag}")
    for hemi, hs in df.groupby("hemi"):
        print(f"\n  【{hemi}】地区数 {hs.region.nunique()}")
        for w, s in hs.groupby("winter"):
            m1, m2 = s.m_s1.median(), s.m_s2.median()
            flag = "" if (m1 >= 5 and m2 >= 5) else f"   ⚠ 对照夏天月数偏少(前 {m1:.0f} / 后 {m2:.0f})"
            print(f"\n    --- {w}  n={len(s)}  冬均温 {s.temp_w.mean():.1f}℃ vs 夏均温 {s.temp_s.mean():.1f}℃"
                  f" (温差 {s.temp_s.mean()-s.temp_w.mean():.1f}℃){flag}")
            for m, nm in (("penalty_cases", "新增确诊(月均)"),
                          ("penalty_deaths", "死亡数(月均)"),
                          ("penalty_deaths_pm", "人均死亡(月均)"),
                          ("penalty_cfr", "病死率")):
                if m not in s.columns:
                    continue
                v = s[m].dropna().to_numpy()
                if len(v) < 6:
                    continue
                wv = stats.wilcoxon(v)
                ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
                print(f"      {nm:16s} {100*(np.exp(v.mean())-1):+7.1f}%  "
                      f"95%CI[{100*(np.exp(ci[0])-1):+6.1f}%,{100*(np.exp(ci[1])-1):+6.1f}%]  "
                      f"更差占比 {100*(v>0).mean():5.1f}%  Wilcoxon p={wv.pvalue:.4g}")
            # 单侧对照分解: 若 pre 与 post 差距极大, 说明"前后夏天平均"这个对照被污染
            for m, nm in (("penalty_deaths", "死亡数"), ("penalty_cfr", "病死率")):
                for suf, lab in (("_pre", "仅对照前一个夏天"), ("_post", "仅对照后一个夏天")):
                    v = s[m + suf].dropna().to_numpy()
                    if len(v) < 6:
                        continue
                    print(f"        └ {nm}·{lab}: {100*(np.exp(v.mean())-1):+7.1f}%  "
                          f"更差占比 {100*(v>0).mean():5.1f}%")


def dose(df, ycol, temp_col="temp_w"):
    out = []
    for hemi, hs in df.groupby("hemi"):
        for w, s in hs.groupby("winter"):
            d = s.dropna(subset=[ycol, temp_col])
            if len(d) < 10:
                continue
            r, p = stats.pearsonr(d[temp_col], d[ycol])
            sr, sp = stats.spearmanr(d[temp_col], d[ycol])
            out.append(dict(hemi=hemi, winter=w, n=len(d), pearson_r=round(r, 3),
                            r2_pct=round(r * r * 100, 1), p=f"{p:.4g}",
                            spearman=round(sr, 3)))
    return pd.DataFrame(out)


if __name__ == "__main__":
    for scope, minc, tag in (("global", 200_000, "全球样本（按半球分开）"),
                             ("us", 100_000, "美国样本（全在北半球）")):
        df = run(scope, minc)
        report(df, tag)
        d = dose(df, "penalty_deaths")
        print(f"\n  [剂量-反应] 冬季气温 vs 死亡惩罚:")
        print(d.to_string(index=False))
        fn = f"hemi_{scope}.csv"
        df.to_csv(fn, index=False, encoding="utf-8-sig")
        d.to_csv(f"hemi_dose_{scope}.csv", index=False, encoding="utf-8-sig")
        print(f"  → 写出 {fn}, hemi_dose_{scope}.csv")
