"""Step 1: 数据质检 —— 量化回调(负增量)规模并验证修复不变量"""
import sys
import numpy as np, pandas as pd
import cw_core as cw

for scope, minc in (("global", 200_000), ("us", 100_000)):
    long, qc = cw.build_panel("region", scope, min_final_cases=minc)
    tag = "全球国家" if scope == "global" else "美国州"
    print(f"\n{'='*70}\n{tag}: {len(qc)} 个地区, 时间 {long.date.min().date()} -> {long.date.max().date()}")
    print(f"  累计确诊合计 {qc.final_confirmed.sum():,.0f}   累计死亡合计 {qc.final_deaths.sum():,.0f}")
    print(f"  回调修复总量: 确诊 {qc.rev_confirmed.sum():,.0f}  死亡 {qc.rev_deaths.sum():,.0f}")
    print(f"  中位回调占比 {qc.rev_pct.median():.3f}%   最大 {qc.rev_pct.max():.2f}% ({qc.loc[qc.rev_pct.idxmax(),'region']})")
    nz = (qc.rev_confirmed + qc.rev_deaths > 0).sum()
    print(f"  发生过回调的地区数: {nz} / {len(qc)}")

    # 不变量1: 修复后单调不减
    bad = 0
    for r, g in long.groupby("region"):
        if (g.sort_values("date")["confirmed"].diff().fillna(0) < -1e-9).any():
            bad += 1
    print(f"  修复后仍存在负增量的地区: {bad} (应为 0)")
    # 不变量2: 终值守恒
    fin = long[long.date == long.date.max()].set_index("region")[["confirmed", "deaths"]]
    j = qc.set_index("region")[["final_confirmed", "final_deaths"]].join(fin, rsuffix="_chk")
    print(f"  终值守恒最大偏差: 确诊 {np.abs(j.final_confirmed-j.confirmed).max():.6f}  死亡 {np.abs(j.final_deaths-j.deaths).max():.6f}")

    print("  --- 回调最严重的 8 个地区 ---")
    top = qc.nlargest(8, "rev_pct")[["region", "lat", "final_confirmed", "rev_confirmed", "rev_deaths", "rev_pct"]]
    print(top.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
    qc.to_csv(f"qc_{scope}.csv", index=False, encoding="utf-8-sig")
