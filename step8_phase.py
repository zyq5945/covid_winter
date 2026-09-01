"""
Step 8: 关键证伪检验 —— "寒冷效应"是否只是"疫情阶段效应"?
============================================================
混杂机制
  一个地区疫情刚暴发时(无检测、无治疗、ICU 挤兑)病死率天然更高; 随着时间推移,
  检测扩大、激素/抗病毒治疗普及、疫苗覆盖、自然感染免疫累积, 病死率系统性下降。
  若某地首次疫情高峰恰好落在寒冷月份(如美国东北部 2020 年 3-4 月),
  而另一地的高峰落在炎热月份(如美国南部 2020 年 6-7 月),
  那么"日历月FE + 地区FE"仍然无法剔除这个混杂 —— 它测到的是**疫情阶段**, 不是**气温**。

控制变量
  C1 log_cum_cases   : 该地区截至当月累计确诊的对数 (疫情推进程度 / 免疫与检测积累)
  C2 region_trend    : 地区专属线性时间趋势 (吸收各地病死率的独立下降路径)
  C3 两者同时加入

判定
  若加入 C1/C2 后 beta 明显趋近 0 且不再显著 -> 原结果是疫情阶段混杂, 不支持寒冷假说。
  若 beta 保持显著为负 -> 寒冷效应独立于疫情阶段, 证据成立。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step4_regression import build_sample, poisson_fe


def prep(scope, minc, lag=21, dfrom="2020-03-01", dto="2022-12-31"):
    g = build_sample(scope, minc, lag, dfrom, dto).sort_values(["region", "date"]).copy()
    g["cum_cases"] = g.groupby("region")["confirmed"].cumsum() - g["confirmed"]
    g["log_cum_cases"] = np.log1p(np.maximum(g["cum_cases"], 0))
    g["t_idx"] = g.groupby("region").cumcount()          # 地区内第几个月
    g["region_trend"] = g["t_idx"] * 1.0
    g["hemi"] = np.where(g["lat"] >= 0, "N", "S")
    g["hemi_month"] = g["hemi"] + "_" + g["cal_month"]
    return g


def fit(g, xcols, fe=("region", "hemi_month"), label=""):
    tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"], x_cols=list(xcols),
                         fe_cols=list(fe), cluster="region")
    r = tab[tab["term"] == "temp_c"]
    if len(r) == 0:
        return None
    r = r.iloc[0]
    return dict(spec=label, beta=r.beta, se=r.se, z=r.z, p=r.p,
                pct_per_10degC=r.pct_effect_per_10degC, n=dg["n"],
                n_clusters=dg["n_clusters"])


def suite(g, tag):
    rows = []
    specs = [
        ("基准 (无阶段控制)", ["temp_c"]),
        ("+ log累计确诊", ["temp_c", "log_cum_cases"]),
        ("+ 地区线性趋势", ["temp_c", "region_trend"]),
        ("+ 两者", ["temp_c", "log_cum_cases", "region_trend"]),
        ("+ 两者 + 累计确诊平方", None),
    ]
    for name, xc in specs:
        if xc is None:
            g = g.copy()
            g["lc2"] = g["log_cum_cases"] ** 2
            xc = ["temp_c", "log_cum_cases", "lc2", "region_trend"]
        r = fit(g, xc, label=name)
        if r:
            r["scope"] = tag
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    allrows = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        g = prep(scope, minc)
        print(f"\n{'='*84}\n{tag}: n={len(g)}, 地区={g.region.nunique()}, "
              f"死亡合计={g.deaths.sum():,.0f}")
        res = suite(g, tag)
        print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        allrows.append(res)

        # 排除 2020 年首轮(检测最不完善、治疗最原始)后再看
        g2 = prep(scope, minc, dfrom="2020-09-01", dto="2022-12-31")
        res2 = suite(g2, tag + " (排除2020前8月)")
        print(f"\n  --- 排除 2020-03~2020-08 首轮 (n={len(g2)}) ---")
        print(res2.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        allrows.append(res2)

    out = pd.concat(allrows, ignore_index=True)
    out.to_csv("phase_confound.csv", index=False, encoding="utf-8-sig")
    print("\n写出 phase_confound.csv")
