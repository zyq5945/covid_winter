"""
Step 4: 双向固定效应识别 —— 排除"变异株/疫苗/检测"时间趋势的混杂
================================================================
为什么需要 FE
  简单的"寒季 vs 暖季"配对无法排除一个致命混杂: 变异株更替与季节转换在时间上重合。
  例: E2021 的寒季(2021-11~2022-03)恰好是奥密克戎 BA.1/BA.2(本身更温和),
      于是"寒季 CFR 更低"其实测到的是"奥密克戎更温和", 而非"冬天不严重"。
  加入 **日历月固定效应 gamma_t** 后, 所有全局时间冲击(变异株严重性、疫苗覆盖、
  检测能力、报告口径)被完全吸收, beta 仅由"同一日历月里, 更冷的地区是否更糟"识别;
  再加入 **地区固定效应 alpha_r**, 所有地区固有特征(医疗水平、年龄结构、纬度、
  报告文化)也被吸收。这是气候-健康文献的标准天气面板设计。

模型 (Poisson FE / 等价于 ppmlhdfe)
  E[deaths_{r,t}] = confirmed_{r,t} * exp(alpha_r + gamma_t + beta * T_{r,t})
  系数含义: 气温每低 1 度, 病死率变化 (exp(beta) - 1) * 100%
  标准误: 按地区聚类稳健
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
from scipy import stats
from scipy.linalg import qr as _qr


def _drop_aliased(X, names, n_keep):
    """
    丢掉与前面 n_keep 列共线的哑变量列, 返回 (X, names, 丢弃列数)。

    为什么需要: 同时放 region FE 与 hemi_month FE 时, 南半球国家会出现
    完全共线 —— "是南半球地区" 这个指示向量既能由所有南半球 region 哑变量相加得到,
    也能由所有 S_* 月份哑变量相加得到(参照组丢的是 N_1, 南半球的月份哑变量一个没丢)。
    结果设计矩阵秩亏: 实测 cond(A) 达 2e17, A^{-1} 元素量级 1e6, 三明治矩阵
    有 37/182 个对角元是负的(最大 -1.7e6) -> np.sqrt 得 NaN, 报
    "invalid value encountered in sqrt", 而且所有涉及列的标准误都不可信。

    做法: 把 FE 块对 x 块做投影残差, 再用列主元 QR 取极大无关组。
    x_cols 一律保留(它们是待估参数, 不可丢)。
    """
    n, k = X.shape
    if k <= n_keep:
        return X, names, 0
    Q, _ = np.linalg.qr(X[:, :n_keep])                       # 投影到 x 块张成的空间
    R = X[:, n_keep:] - Q @ (Q.T @ X[:, n_keep:])            # FE 块残差
    _, rr, piv = _qr(R, mode="economic", pivoting=True)
    d = np.abs(np.diag(rr))
    tol = (d[0] if d.size else 0.0) * max(n, k) * np.finfo(float).eps * 1e3
    rank = int((d > max(tol, 0.0)).sum())
    if rank == k - n_keep:
        return X, names, 0
    sel = sorted(piv[:rank].tolist())                        # 保持原顺序, 便于解读
    keep_cols = list(range(n_keep)) + [n_keep + j for j in sel]
    return X[:, keep_cols], [names[i] for i in keep_cols], (k - n_keep) - rank

from step3_analysis import monthly_panel, DEFAULT_LAG, MIN_MONTH_CASES


# ------------------------------------------------------------------ Poisson FE
def poisson_fe(df, y, offset_cols, x_cols, fe_cols, cluster,
               max_iter=100, tol=1e-10, ridge=1e-8):
    """
    Poisson 固定效应 (IRLS)。offset 取对数后作为固定偏移。
    y: 死亡数; offset_cols: 用作暴露量的列(取 log 后相加进线性预测)
    返回 (coef 表, 诊断 dict)
    """
    d = df.dropna(subset=[y] + x_cols + offset_cols + fe_cols + [cluster]).copy()
    yy = d[y].to_numpy(float)
    log_off = np.zeros(len(d))
    for c in offset_cols:
        v = d[c].to_numpy(float)
        if (v <= 0).any():
            d = d[v > 0]
            yy = d[y].to_numpy(float)
            v = d[c].to_numpy(float)
            log_off = np.zeros(len(d))
        log_off = log_off + np.log(v)

    names, blocks = list(x_cols), [d[c].to_numpy(float) for c in x_cols]
    for f in fe_cols:
        codes, uniq = pd.factorize(d[f])
        M = np.zeros((len(d), len(uniq)))
        M[np.arange(len(d)), codes] = 1.0
        M = M[:, 1:]                      # 丢第一类作参照
        blocks.append(M)
        names += [f"{f}={u}" for u in uniq[1:]]
    X = np.column_stack(blocks)
    X, names, n_aliased = _drop_aliased(X, names, len(x_cols))
    n, k = X.shape

    beta = np.zeros(k)
    beta[:len(x_cols)] = 0.0
    mu_off = log_off.copy()
    for it in range(max_iter):
        eta = mu_off + X @ beta
        eta = np.clip(eta, -30, 30)
        mu = np.exp(eta)
        z = eta - mu_off + (yy - mu) / np.maximum(mu, 1e-12)   # working response
        W = np.maximum(mu, 1e-12)
        XtW = X.T * W
        A = XtW @ X + ridge * np.eye(k)
        b_new = np.linalg.solve(A, XtW @ z)
        if np.max(np.abs(b_new - beta)) < tol:
            beta = b_new
            break
        beta = b_new

    eta = np.clip(mu_off + X @ beta, -30, 30)
    mu = np.exp(eta)
    resid = yy - mu

    # 聚类稳健三明治
    A = (X.T * np.maximum(mu, 1e-12)) @ X + ridge * np.eye(k)
    cl = pd.factorize(d[cluster])[0]
    meat = np.zeros((k, k))
    for g in np.unique(cl):
        m = cl == g
        u = X[m].T @ resid[m]
        meat += np.outer(u, u)
    G = len(np.unique(cl))
    corr = G / max(G - 1, 1)

    keep = list(range(len(x_cols)))
    # 只解出待估项对应的 A^{-1} 行, 不必构造完整的 k×k 方差矩阵(k 可达上百)。
    # clip: V 理论上半正定, 浮点误差会让对角元出现 -1e-16 量级的负值 -> sqrt 得 NaN。
    Sel = np.eye(k)[keep]                    # (m, k) 选择矩阵
    B = np.linalg.solve(A, Sel.T).T          # = A^{-1}[keep, :]
    dv = np.diag(corr * B @ meat @ B.T)
    n_neg = int((dv < 0).sum())
    se = np.full(k, np.nan)
    se[keep] = np.sqrt(np.clip(dv, 0.0, None))

    tab = pd.DataFrame({
        "term": [names[i] for i in keep],
        "beta": beta[keep],
        "se": se[keep],
        "z": beta[keep] / se[keep],
        "p": 2 * stats.norm.sf(np.abs(beta[keep] / se[keep])),
    })
    tab["pct_effect_per_degC"] = 100.0 * (np.exp(tab["beta"]) - 1.0)
    # 10 度降温的效应
    tab["pct_effect_per_10degC"] = 100.0 * (np.exp(10 * tab["beta"]) - 1.0)
    # 伪 R2 (相对只有 offset 的截距模型)
    ll = float(np.sum(yy * np.log(np.maximum(mu, 1e-12)) - mu))
    diag = dict(n=n, k=k, n_clusters=G, loglik=ll, iters=it + 1,
                mean_deaths=float(yy.mean()), total_deaths=float(yy.sum()),
                cond_A=float(np.linalg.cond(A)), n_aliased=n_aliased,
                n_neg_var=n_neg)
    return tab, diag


# ------------------------------------------------------------------ 样本构造
def build_sample(scope, minc, lag, date_from=None, date_to=None, min_month_cases=MIN_MONTH_CASES,
                 temp_path="daily_temperature.csv"):
    import cw_core as cw
    long, qc = cw.build_panel("region", scope, min_final_cases=minc)
    pop = qc.set_index("region")["population"] if scope == "us" else None
    g = monthly_panel(long, lag, pop)
    tp = pd.read_csv(temp_path)
    tp["date"] = pd.to_datetime(tp["date"])
    tp = tp[tp["scope"] == scope]
    tmon = tp.groupby(["region", tp["date"].values.astype("datetime64[M]")], as_index=False)["temp_c"].mean()
    tmon.columns = ["region", "ym", "temp_c"]
    tmon["ym"] = pd.to_datetime(tmon["ym"])
    g = g.merge(tmon, on=["region", "ym"], how="left")
    clim = g.groupby("region")["temp_c"].agg(["min", "max"])
    clim["amplitude"] = clim["max"] - clim["min"]
    g = g.merge(clim[["amplitude"]], on="region", how="left")

    g["date"] = pd.to_datetime(g["ym"])
    if date_from:
        g = g[g["date"] >= pd.Timestamp(date_from)]
    if date_to:
        g = g[g["date"] <= pd.Timestamp(date_to)]
    g = g.dropna(subset=["temp_c"])
    g = g[(g["confirmed"] >= min_month_cases) & (g["deaths"] > 0)]
    g["cal_month"] = g["date"].dt.strftime("%Y-%m")
    # 温带/寒带标记: 年均温
    g["annual_t"] = g.groupby("region")["temp_c"].transform("mean")
    return g


def run_specs(g, tag, label):
    rows, diags = [], []
    specs = [
        ("A 地区FE + 日历月FE (最严格)", ["region"], ["cal_month"]),
        ("B 地区FE + 流行年FE", ["region"], ["epoch"]),
        ("C 仅地区FE", ["region"], []),
    ]
    for name, fe1, fe2 in specs:
        fe = fe1 + fe2
        tab, dg = poisson_fe(g, y="deaths", offset_cols=["confirmed"],
                             x_cols=["temp_c"], fe_cols=fe, cluster="region")
        r = tab.iloc[0]
        rows.append(dict(spec=name, scope=tag, sample=label,
                         beta=r.beta, se=r.se, z=r.z, p=r.p,
                         pct_per_degC=r.pct_effect_per_degC,
                         pct_per_10degC=r.pct_effect_per_10degC,
                         n=dg["n"], n_clusters=dg["n_clusters"]))
        diags.append(dg)
    return pd.DataFrame(rows), diags


if __name__ == "__main__":
    allrows = []
    for scope, minc, tag in (("us", 100_000, "美国州"), ("global", 200_000, "全球国家")):
        for label, df_, dt_ in (("全期 2020-03~2022-12", "2020-03-01", "2022-12-31"),
                                ("疫苗前 2020-03~2021-10", "2020-03-01", "2021-10-31")):
            g = build_sample(scope, minc, DEFAULT_LAG, df_, dt_)
            if len(g) < 200:
                print(f"  样本过小跳过: {tag} {label} n={len(g)}")
                continue
            n_reg = g["region"].nunique()
            print(f"\n{'='*78}\n{tag} | {label} | lag={DEFAULT_LAG} | "
                  f"{len(g)} 个地区-月, {n_reg} 个地区, 死亡合计 {g.deaths.sum():,.0f}")
            res, dg = run_specs(g, tag, label)
            print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
            allrows.append(res)
            g.to_csv(f"panel_{scope}_{label[:4]}.csv", index=False, encoding="utf-8-sig")
    out = pd.concat(allrows, ignore_index=True)
    out.to_csv("fe_regressions.csv", index=False, encoding="utf-8-sig")
    print("\n写出 fe_regressions.csv")
