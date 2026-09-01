"""
Step 3: 主分析 —— 寒冷是否让 COVID-19 死亡结局更糟
==================================================
三层证据链
  L1 时序内对照: 每个地区在 **同一个流行年 (epoch = 9月1日~次年8月31日)** 内部,
                 比较寒季月 vs 暖季月的滞后对齐 CFR。同一 epoch 内比较可同时
                 控制疫苗覆盖、变异株、检测能力等随时间演化的混杂。
  L2 配对检验  : 地区层面寒季劣势的中位数、符号检验、Wilcoxon 符号秩检验。
  L3 剂量-反应 : 双向固定效应回归 (地区 FE + 时间 FE), 用**实测气温**而非纬度代理,
                 并检验"季节温差幅度越大的地区, 寒季劣势越大"。

滞后对齐 CFR (lag-aligned CFR)
  CFR_m = (第 m 月感染者在 m 月后 L 天内死亡的人数) / (第 m 月新增确诊)
  即 deaths over [start_m + L, end_m + L] / confirmed over month m。
  默认 L=21 天, 稳健性扫描 L ∈ {14, 21, 28}。
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd

from scipy import stats

LAGS = [14, 21, 28]
DEFAULT_LAG = 21
MIN_MONTH_CASES = 500          # 月确诊下限, 低于此的月不进入 CFR 计算
MIN_EPOCH_CASES = 20_000       # 地区-epoch 确诊下限

# epoch: 9/1(year) -> 8/31(year+1); 寒季 = 11,12,1,2,3 月; 暖季 = 9,5,6,7,8 月
COLD_MONTHS_NH = {11, 12, 1, 2, 3}
WARM_MONTHS_NH = {9, 5, 6, 7, 8}


def epoch_of(ts):
    """返回流行年标签: 2020-09-01 ~ 2021-08-31 -> 'E2020'"""
    y = ts.year if ts.month >= 9 else ts.year - 1
    return f"E{y}"


def season_nh(month):
    if month in COLD_MONTHS_NH:
        return "cold"
    if month in WARM_MONTHS_NH:
        return "warm"
    return "transition"


# ------------------------------------------------------------------ 面板
def monthly_panel(long: pd.DataFrame, lag: int, pop: pd.Series | None = None) -> pd.DataFrame:
    """按月聚合 + 滞后对齐死亡归因"""
    df = long.sort_values(["region", "date"]).copy()
    last_date = df["date"].max()

    df["attr_deaths"] = df.groupby("region")["new_deaths"].shift(-lag)
    df["ym"] = df["date"].values.astype("datetime64[M]")
    g = df.groupby(["region", "ym"], as_index=False).agg(
        confirmed=("new_confirmed", "sum"),
        deaths=("attr_deaths", "sum"),
        raw_deaths=("new_deaths", "sum"),
        days=("date", "count"),
    )
    # 死亡窗口完整性: 该月月末 + lag 必须落在数据内
    month_end = pd.to_datetime(g["ym"]) + pd.offsets.MonthEnd(0)
    g = g[(month_end + pd.Timedelta(days=lag)) <= last_date]
    # 月份完整性(排除首尾残缺月)
    g["dim"] = month_end.dt.days_in_month
    g = g[g["days"] >= g["dim"] - 1]

    g["month"] = pd.to_datetime(g["ym"]).dt.month
    g["lat"] = g["region"].map(long.groupby("region")["lat"].first())
    g["south"] = g["lat"] < 0
    g["season"] = [("cold" if season_nh(m) == "cold" else "warm" if season_nh(m) == "warm" else "transition")
                   if not s else
                   ("warm" if season_nh(m) == "cold" else "cold" if season_nh(m) == "warm" else "transition")
                   for m, s in zip(g["month"], g["south"])]
    g["epoch"] = [epoch_of(pd.Timestamp(x)) for x in g["ym"]]
    if pop is not None:
        g["pop"] = g["region"].map(pop)
        g["deaths_pm"] = 1e6 * g["deaths"] / g["pop"]
        g["cases_pm"] = 1e6 * g["confirmed"] / g["pop"]
    g["cfr"] = np.where(g["confirmed"] >= MIN_MONTH_CASES, g["deaths"] / g["confirmed"], np.nan)
    return g


# ------------------------------------------------------------------ L1 配对
def epoch_pairs(g: pd.DataFrame):
    """每个 (region, epoch) 聚合出寒季/暖季的合并 CFR, 并给出配对"""
    gg = g[g["season"].isin(["cold", "warm"])].copy()
    agg = gg.groupby(["region", "epoch", "season"], as_index=False).agg(
        confirmed=("confirmed", "sum"), deaths=("deaths", "sum"),
        months=("ym", "count"))
    agg = agg[agg["confirmed"] > 0]
    agg["cfr"] = agg["deaths"] / agg["confirmed"]
    piv = agg.pivot_table(index=["region", "epoch"], columns="season",
                          values=["confirmed", "deaths", "cfr", "months"]).reset_index()
    piv.columns = ["_".join([c for c in col if c]) for col in piv.columns]
    need = ["cfr_cold", "cfr_warm", "confirmed_cold", "confirmed_warm", "months_cold", "months_warm"]
    for c in need:
        if c not in piv.columns:
            piv[c] = np.nan
    piv = piv.dropna(subset=need[:2])
    piv = piv[(piv["confirmed_cold"] >= MIN_EPOCH_CASES) & (piv["confirmed_warm"] >= MIN_EPOCH_CASES)]
    piv = piv[(piv["months_cold"] >= 3) & (piv["months_warm"] >= 3)]
    piv["cfr_ratio"] = piv["cfr_cold"] / piv["cfr_warm"]
    return piv


def paired_report(piv: pd.DataFrame, label: str, by_epoch: bool = True):
    out = []
    grp = piv.groupby("epoch") if by_epoch else [("ALL", piv)]
    for ep, sub in grp:
        if len(sub) < 5:
            continue
        r = sub["cfr_ratio"].to_numpy()
        n_cold_worse = int((r > 1).sum())
        w = stats.wilcoxon(np.log(r), alternative="two-sided")
        s = stats.binomtest(n_cold_worse, len(r), 0.5, alternative="greater")
        out.append(dict(
            scope=label, epoch=ep, n_regions=len(r),
            n_cold_worse=n_cold_worse, pct_cold_worse=100.0 * n_cold_worse / len(r),
            median_ratio=float(np.median(r)),
            geo_mean_ratio=float(np.exp(np.mean(np.log(r)))),
            ci_lo=float(np.exp(np.mean(np.log(r)) - 1.96 * stats.sem(np.log(r)))),
            ci_hi=float(np.exp(np.mean(np.log(r)) + 1.96 * stats.sem(np.log(r)))),
            wilcoxon_p=float(w.pvalue), sign_p=float(s.pvalue),
            median_cfr_cold=float(sub["cfr_cold"].median() * 100),
            median_cfr_warm=float(sub["cfr_warm"].median() * 100),
        ))
    return pd.DataFrame(out)


# ------------------------------------------------------------------ L3 固定效应
def demean(mat, codes, ngroups, weights):
    """按 codes 做加权去均值"""
    out = mat.copy()
    wsum = np.bincount(codes, weights=weights, minlength=ngroups)
    wsum[wsum == 0] = 1.0
    for j in range(mat.shape[1]):
        s = np.bincount(codes, weights=weights * mat[:, j], minlength=ngroups)
        out[:, j] = mat[:, j] - (s / wsum)[codes]
    return out


def two_way_fe(df, y_col, x_cols, fe1, fe2, weights, cluster, n_iter=200, tol=1e-10):
    """
    双向固定效应加权最小二乘。用交替投影吸收两个维度的 FE。
    返回 (beta, se_cluster, n, r2_within)
    """
    d = df.dropna(subset=[y_col] + x_cols + [weights]).copy()
    y = d[y_col].to_numpy(float)
    X = d[x_cols].to_numpy(float)
    if "const" not in x_cols:
        X = np.column_stack([np.ones(len(d)), X])
        names = ["const"] + list(x_cols)
    else:
        names = list(x_cols)
    w = d[weights].to_numpy(float)
    c1 = pd.factorize(d[fe1])[0]
    c2 = pd.factorize(d[fe2])[0]
    cl = pd.factorize(d[cluster])[0]
    n1, n2 = c1.max() + 1, c2.max() + 1

    sw = np.sqrt(w)
    Y, Xw = y * sw, X * sw[:, None]

    def absorb(M):
        prev = None
        for _ in range(n_iter):
            M = demean(M, c1, n1, w)
            M = demean(M, c2, n2, w)
            if prev is not None and np.max(np.abs(M - prev)) < tol:
                break
            prev = M.copy()
        return M

    Yd, Xd = absorb(Y[:, None])[:, 0], absorb(Xw)
    # 加权 LS（已乘 sqrt(w)，故为 OLS）
    XtX = Xd.T @ Xd
    beta = np.linalg.solve(XtX, Xd.T @ Yd)
    resid = Yd - Xd @ beta

    # 聚类稳健标准误（对 FE 吸收的自由度做简单校正）
    XtXinv = np.linalg.inv(XtX)
    meat = np.zeros((Xd.shape[1], Xd.shape[1]))
    for g in np.unique(cl):
        m = cl == g
        u = Xd[m].T @ resid[m]
        meat += np.outer(u, u)
    G, N, K = len(np.unique(cl)), len(d), Xd.shape[1] + n1 + n2
    corr = (G / (G - 1.0)) * ((N - 1.0) / (N - K))
    V = corr * XtXinv @ meat @ XtXinv
    se = np.sqrt(np.diag(V))

    ss_tot = float(np.sum(Yd ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else np.nan
    res = pd.DataFrame({"term": names, "beta": beta, "se": se,
                        "t": beta / se,
                        "p": 2 * stats.norm.sf(np.abs(beta / se))})
    return res, dict(n=N, n_groups=G, r2_within=r2, n_fe1=n1, n_fe2=n2)


# ------------------------------------------------------------------ 主流程
def run(scope, minc, tag, temp_path="daily_temperature.csv"):
    import cw_core as cw
    print(f"\n{'='*78}\n{tag}\n{'='*78}")
    long, qc = cw.build_panel("region", scope, min_final_cases=minc)
    pop = None
    if scope == "us":
        pop = qc.set_index("region")["population"]

    # 合并气温
    tp = pd.read_csv(temp_path)
    tp["date"] = pd.to_datetime(tp["date"])
    tp = tp[tp["scope"] == scope]
    tmon = tp.groupby(["region", tp["date"].values.astype("datetime64[M]")], as_index=False)["temp_c"].mean()
    tmon = tmon.rename(columns={"temp_c": "temp_c", "date": "ym"})
    tmon.columns = ["region", "ym", "temp_c"]
    tmon["ym"] = pd.to_datetime(tmon["ym"])

    results = {}
    all_pairs = []
    for lag in LAGS:
        g = monthly_panel(long, lag, pop)
        g = g.merge(tmon, on=["region", "ym"], how="left")
        # 季节温差幅度 / 冬季气温 (按地区, 用全期月均温)
        clim = g.groupby("region")["temp_c"].agg(["min", "max", "mean"])
        clim["amplitude"] = clim["max"] - clim["min"]
        g = g.merge(clim[["amplitude"]], on="region", how="left")
        g["lag"] = lag
        piv = epoch_pairs(g)
        if pop is not None:
            pm = g.groupby(["region", "epoch", "season"], as_index=False).agg(
                deaths=("deaths", "sum"), pop=("pop", "first"))
            pm["dpm"] = 1e6 * pm["deaths"] / pm["pop"]
            pmp = pm.pivot_table(index=["region", "epoch"], columns="season", values="dpm")
            pmp.columns = [f"dpm_{c}" for c in pmp.columns]
            piv = piv.merge(pmp, left_on=["region", "epoch"], right_index=True, how="left")
            piv["dpm_ratio"] = piv["dpm_cold"] / piv["dpm_warm"]
        piv["lag"] = lag
        all_pairs.append(piv)
        rep = paired_report(piv, tag)
        if lag == DEFAULT_LAG:
            print(f"\n[lag={lag}] 地区内配对: 寒季 CFR / 暖季 CFR")
            print(rep.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        results[lag] = (g, piv, rep)

    pairs = pd.concat(all_pairs, ignore_index=True)
    return long, qc, results, pairs


def robust_summary(pairs, tag):
    rows = []
    for lag, sub in pairs.groupby("lag"):
        for ep, s in sub.groupby("epoch"):
            if len(s) < 5:
                continue
            r = s["cfr_ratio"].to_numpy()
            w = stats.wilcoxon(np.log(r))
            rows.append(dict(scope=tag, lag=lag, epoch=ep, n=len(r),
                             median_ratio=float(np.median(r)),
                             pct_cold_worse=100.0 * (r > 1).mean(),
                             wilcoxon_p=float(w.pvalue)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = {}
    for scope, minc, tag in (("us", 100_000, "美国 50 州+DC"), ("global", 200_000, "全球国家")):
        long, qc, results, pairs = run(scope, minc, tag)
        pairs.to_csv(f"pairs_{scope}.csv", index=False, encoding="utf-8-sig")
        rs = robust_summary(pairs, tag)
        print(f"\n--- 多 lag 稳健性: {tag} ---")
        print(rs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        out[scope] = (long, qc, results, pairs, rs)
        results[DEFAULT_LAG][0].to_csv(f"monthly_{scope}.csv", index=False, encoding="utf-8-sig")
    print("\n完成。")
