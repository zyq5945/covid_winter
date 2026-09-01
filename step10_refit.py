"""Step 10: 用修正坐标后的气温重估跨国样本, 并与美国样本并列"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step4_regression import build_sample, poisson_fe
from step8_phase import prep

V2 = "daily_temperature_v2.csv"
V1 = "daily_temperature.csv"


def spec(g, fe=("region", "hemi_month")):
    tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"], x_cols=["temp_c"],
                         fe_cols=list(fe), cluster="region")
    r = tab.iloc[0]
    return dict(beta=r.beta, se=r.se, p=r.p, pct_per_10degC=r.pct_effect_per_10degC,
                n=dg["n"], n_clusters=dg["n_clusters"])


def prep_tp(scope, minc, lag=21, dfrom="2020-03-01", dto="2022-12-31", tp=V2):
    g = build_sample(scope, minc, lag, dfrom, dto, temp_path=tp).sort_values(["region", "date"]).copy()
    g["cum_cases"] = g.groupby("region")["confirmed"].cumsum() - g["confirmed"]
    g["log_cum_cases"] = np.log1p(np.maximum(g["cum_cases"], 0))
    g["t_idx"] = g.groupby("region").cumcount()
    g["region_trend"] = g["t_idx"] * 1.0
    g["hemi"] = np.where(g["lat"] >= 0, "N", "S")
    g["hemi_month"] = g["hemi"] + "_" + g["cal_month"]
    return g


if __name__ == "__main__":
    rows = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        for tpname, tp in (("原坐标", V1), ("修正坐标", V2)):
            for label, df_, dt_ in (("全期 2020-03~2022-12", "2020-03-01", "2022-12-31"),
                                    ("剔除首轮 2020-09~2022-12", "2020-09-01", "2022-12-31"),
                                    ("疫苗前 2020-09~2021-10", "2020-09-01", "2021-10-31")):
                g = prep_tp(scope, minc, 21, df_, dt_, tp)
                if g["region"].nunique() < 10 or len(g) < 200:
                    continue
                r = spec(g)
                r.update(scope=tag, coords=tpname, sample=label,
                         n_regions=g["region"].nunique())
                rows.append(r)

    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["scope", "sample"], columns="coords",
                          values=["beta", "p", "pct_per_10degC"])
    print("\n【坐标修正前后对比】Spec A': 地区FE + 半球x日历月FE, lag=21")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    out.to_csv("coord_fix.csv", index=False, encoding="utf-8-sig")

    # 修正坐标后, 跨国样本的纬度带 + 阶段控制
    print("\n【修正坐标后】全球国家: 阶段控制 + 纬度带")
    g = prep_tp("global", 200_000, 21, "2020-03-01", "2022-12-31", V2)
    extra = []
    for label, xc in (("基准", ["temp_c"]),
                      ("+log累计确诊", ["temp_c", "log_cum_cases"]),
                      ("+地区趋势", ["temp_c", "region_trend"]),
                      ("+两者", ["temp_c", "log_cum_cases", "region_trend"])):
        tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"], x_cols=xc,
                             fe_cols=["region", "hemi_month"], cluster="region")
        r = tab[tab.term == "temp_c"].iloc[0]
        extra.append(dict(spec=label, beta=r.beta, se=r.se, p=r.p,
                          pct_per_10degC=r.pct_effect_per_10degC, n=dg["n"]))
    print(pd.DataFrame(extra).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    band = []
    for lo, hi, nm in ((0, 23.5, "热带 |lat|<23.5"), (23.5, 40, "亚热带 23.5-40"), (40, 90, "温带寒带 |lat|>=40")):
        sub = g[(g["lat"].abs() >= lo) & (g["lat"].abs() < hi)]
        if sub["region"].nunique() < 8:
            continue
        r = spec(sub)
        r.update(band=nm, n_regions=sub["region"].nunique())
        band.append(r)
    print("\n  分纬度带:")
    print(pd.DataFrame(band)[["band", "n_regions", "n", "beta", "se", "p", "pct_per_10degC"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pd.DataFrame(band).to_csv("global_bands_v2.csv", index=False, encoding="utf-8-sig")
