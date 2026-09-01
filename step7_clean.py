"""
Step 7: 最干净设定 + 跨国样本为何失效
=====================================
问题: 单一"日历月FE"把北半球冬季的全局冲击吸收掉了, 而南半球国家在同一日历月是夏季,
      其 FE 残差把"南半球夏季"和"不在北半球冬季"混在一起, 识别不干净。
改进: 用 **半球 x 日历月 FE**, 识别仅来自"同一半球、同一日历月内, 更冷的地区是否更糟"。

同时对跨国样本做数据质量筛查:
  - 累计 CFR 落在合理区间(死亡报告完整性 & 检测强度的综合体检)
  - 排除 2022 年(奥密克戎时代家庭自测导致确诊严重漏报, CFR 分母失真)
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats

from step4_regression import build_sample, poisson_fe

QUAL_CFR_LO, QUAL_CFR_HI = 0.002, 0.045      # 累计 CFR 合理区间


def spec(g, fe_time, xcol="temp_c"):
    tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"], x_cols=[xcol],
                         fe_cols=["region"] + list(fe_time), cluster="region")
    r = tab.iloc[0]
    return dict(beta=r.beta, se=r.se, z=r.z, p=r.p,
                pct_per_10degC=r.pct_effect_per_10degC,
                n=dg["n"], n_clusters=dg["n_clusters"])


if __name__ == "__main__":
    rows = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        g = build_sample(scope, minc, 21, "2020-03-01", "2022-12-31")
        g["hemi"] = np.where(g["lat"] >= 0, "N", "S")
        g["hemi_month"] = g["hemi"] + "_" + g["cal_month"]

        print(f"\n{'='*82}\n{tag}: n={len(g)}, 地区={g.region.nunique()}")

        variants = [
            ("日历月FE", ["cal_month"], g),
            ("半球x日历月FE (最干净)", ["hemi_month"], g),
            ("半球x日历月FE + 排除2022", ["hemi_month"], g[g["cal_month"] < "2022-01"]),
        ]
        for name, fe, sub in variants:
            if sub["region"].nunique() < 10 or len(sub) < 200:
                continue
            r = spec(sub, fe)
            r.update(scope=tag, spec=name)
            rows.append(r)
            print(f"  {name:28s} n={r['n']:5d} beta={r['beta']:+.4f} se={r['se']:.4f} "
                  f"p={r['p']:.4g}  每降10C: {r['pct_per_10degC']:+.1f}%")

        # --- 跨国数据质量筛查 ---
        if scope == "global":
            qc = g.groupby("region").agg(cases=("confirmed", "sum"), deaths=("deaths", "sum"))
            qc["cfr"] = qc["deaths"] / qc["cases"]
            print(f"\n  累计 CFR 分布: p10={qc.cfr.quantile(.1):.4f} 中位={qc.cfr.median():.4f} "
                  f"p90={qc.cfr.quantile(.9):.4f}")
            good = qc[(qc.cfr >= QUAL_CFR_LO) & (qc.cfr <= QUAL_CFR_HI)].index
            print(f"  CFR 落在 [{QUAL_CFR_LO:.3f}, {QUAL_CFR_HI:.3f}] 的国家: "
                  f"{len(good)}/{len(qc)}  (被剔除: "
                  f"{', '.join(sorted(set(qc.index) - set(good))[:14])})")
            for nm, sub in (("质量筛查后(半球x月FE)", g[g.region.isin(good)]),
                            ("质量筛查 + 排除2022", g[g.region.isin(good) & (g.cal_month < "2022-01")])):
                if sub["region"].nunique() < 10:
                    continue
                r = spec(sub, ["hemi_month"])
                r.update(scope=tag, spec=nm)
                rows.append(r)
                print(f"  {nm:28s} n={r['n']:5d} beta={r['beta']:+.4f} se={r['se']:.4f} "
                      f"p={r['p']:.4g}  每降10C: {r['pct_per_10degC']:+.1f}%")

            # 分纬度带
            for lo, hi, nm in ((0, 23.5, "热带 |lat|<23.5"), (23.5, 40, "亚热带 23.5-40"),
                               (40, 90, "温带/寒带 |lat|>=40")):
                sub = g[(g["lat"].abs() >= lo) & (g["lat"].abs() < hi)]
                if sub["region"].nunique() < 8:
                    continue
                r = spec(sub, ["hemi_month"])
                r.update(scope=tag, spec=nm)
                rows.append(r)
                print(f"  {nm:28s} n={r['n']:5d} beta={r['beta']:+.4f} se={r['se']:.4f} "
                      f"p={r['p']:.4g}  每降10C: {r['pct_per_10degC']:+.1f}%")

    out = pd.DataFrame(rows)
    out.to_csv("spec_robustness.csv", index=False, encoding="utf-8-sig")
    print("\n写出 spec_robustness.csv")
