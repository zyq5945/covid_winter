"""
COVID-19 冬季寒冷假说 —— 公共模块
---------------------------------
职责：加载 JHU CSSE 时间序列 -> 地区聚合 -> 数据修订(回调)修复 -> 7 日平滑 -> 波次分层 -> 寒暖季标注

设计要点
1. 回调修复 (revision repair):
   JHU 累计序列存在向下修订(如某国剔除重复病例、美国州更换统计口径)，
   表现为 c[t] < c[t-1]。直接 diff 会产生负增量，使"阶段新增"失真。
   采用 **回溯剥离法 (backward carry-back)**：将第 t 天的负增量 m 沿时间轴向后
   摊销到最近的正增量上，直到 m 被扣完。性质：
     - 修复后序列单调不减
     - 终值严格等于原始终值 (总量守恒)
     - 只改动修订点之前的近期日子，不污染远期历史
2. 波次分层 (wave stratification):
   在 7 日平滑新增序列上用相对阈值法切波，避免把"冬季一波 vs 夏季一波"
   混在同一个阶段里比较。
3. 季节标注:
   北半球 11/12/1/2/3 月 = 寒季, 5-9 月 = 暖季; 南半球按月份取反;
   4/10 月为过渡月, 单独标记, 主分析中剔除。
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

DATA_DIR = r"D:/q/work/my/md/xs/COVID-19-master/csse_covid_19_data/csse_covid_19_time_series_ex"

# ---------------------------------------------------------------- 季节定义
# 北半球：寒季(冷) 11,12,1,2,3；暖季(热) 5,6,7,8,9；过渡 4,10
NH_COLD = {11, 12, 1, 2, 3}
NH_WARM = {5, 6, 7, 8, 9}


def season_of(month: int, lat: float) -> str:
    """按月份与纬度所在半球返回 'cold' / 'warm' / 'transition'"""
    southern = lat < 0
    cold = (NH_COLD if not southern else NH_WARM)
    warm = (NH_WARM if not southern else NH_COLD)
    if month in cold:
        return "cold"
    if month in warm:
        return "warm"
    return "transition"


# ---------------------------------------------------------------- 回调修复
def repair_monotone(cum: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    回溯剥离法把累计序列修成单调不减，并保持终值不变。

    返回 (repaired_cum, repaired_inc, total_revision)
      total_revision = 所有负增量绝对值之和 (相对终值的修订强度)
    """
    c = np.asarray(cum, dtype=np.float64).copy()
    n = len(c)
    inc = np.zeros(n)
    if n:
        inc[0] = c[0]
    total_rev = 0.0

    for t in range(1, n):
        d = c[t] - c[t - 1]
        if d >= 0:
            inc[t] = d
            continue
        # 负增量：向下修订 m
        m = -d
        total_rev += m
        inc[t] = 0.0
        # 沿时间轴向后摊销到最近的正增量
        k = t - 1
        while m > 1e-9 and k >= 0:
            take = min(inc[k], m)
            inc[k] -= take
            m -= take
            k -= 1
        if m > 1e-9:          # 历史增量不够扣(极罕见)，压到 t=0 的初值上
            inc[0] = max(inc[0] - m, 0.0)
    repaired = np.cumsum(inc)
    return repaired, inc, total_rev


# ---------------------------------------------------------------- 加载
def _split_date_cols(cols):
    return [c for c in cols if "/" in c and c.count("/") == 2]


def _to_dates(cols):
    return pd.to_datetime(cols, format="%m/%d/%y")


def load_global(kind: str) -> pd.DataFrame:
    """kind in {'confirmed','deaths','recovered'} -> 长表 [region, date, value]"""
    path = os.path.join(DATA_DIR, f"time_series_covid19_{kind}_global.csv")
    df = pd.read_csv(path)
    dcols = _split_date_cols(df.columns)
    dates = _to_dates(dcols)
    meta = ["Province/State", "Country/Region", "Lat", "Long"]
    m = df.melt(id_vars=meta, value_vars=dcols, var_name="date", value_name="value")
    m["date"] = pd.to_datetime(m["date"].map(dict(zip(dcols, dates))))
    m["value"] = pd.to_numeric(m["value"], errors="coerce").fillna(0.0)
    return m


def load_us(kind: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"time_series_covid19_{kind}_US.csv")
    df = pd.read_csv(path)
    dcols = _split_date_cols(df.columns)
    dates = _to_dates(dcols)
    meta = ["Province_State", "Lat", "Long_"]
    if "Population" in df.columns:
        meta.append("Population")
    m = df.melt(id_vars=meta, value_vars=dcols, var_name="date", value_name="value")
    m["date"] = pd.to_datetime(m["date"].map(dict(zip(dcols, dates))))
    m["value"] = pd.to_numeric(m["value"], errors="coerce").fillna(0.0)
    return m


# ---------------------------------------------------------------- 面板构建
def build_panel(region_col: str, scope: str, min_final_cases: int = 0):
    """
    返回 (long, wide, qc)
      long: [region, date, confirmed, deaths, recovered(可空)]
      wide: index=region, columns=date 的累计矩阵 (已修复)
      qc  : [region, lat, ..., rev_confirmed, rev_deaths, rev_pct]
    """
    if scope == "global":
        conf = load_global("confirmed").groupby(["Country/Region", "date"], as_index=False)["value"].sum()
        conf = conf.rename(columns={"Country/Region": "region"})
        dead = load_global("deaths").groupby(["Country/Region", "date"], as_index=False)["value"].sum()
        dead = dead.rename(columns={"Country/Region": "region"})
        try:
            rec = load_global("recovered").groupby(["Country/Region", "date"], as_index=False)["value"].sum()
            rec = rec.rename(columns={"Country/Region": "region"})
        except FileNotFoundError:
            rec = None
        # 病例加权平均经纬度：用各省终值病例数加权，避免海外飞地拉偏
        # 注意：部分行(如 Canada/Repatriated Travellers, China/Unknown)坐标为 NaN，
        #       必须剔除后再加权，否则 np.average 会把整个国家的中心污染成 NaN。
        lat_src = load_global("confirmed")
        lat_src = lat_src[lat_src["date"] == lat_src["date"].max()]
        lat_src = lat_src[np.isfinite(lat_src["Lat"]) & np.isfinite(lat_src["Long"])]
        lat_src = lat_src.assign(_w=np.maximum(lat_src["value"], 0) + 1e-9)

        def _wavg(g, col):
            return float(np.average(g[col], weights=g["_w"])) if len(g) else np.nan

        lat_map = lat_src.groupby("Country/Region").apply(_wavg, "Lat", include_groups=False).to_dict()
        lon_map = lat_src.groupby("Country/Region").apply(_wavg, "Long", include_groups=False).to_dict()
    else:  # US states
        conf = load_us("confirmed").groupby(["Province_State", "date"], as_index=False)["value"].sum()
        conf = conf.rename(columns={"Province_State": "region"})
        dead = load_us("deaths").groupby(["Province_State", "date"], as_index=False)["value"].sum()
        dead = dead.rename(columns={"Province_State": "region"})
        rec = None
        lat_src = load_us("confirmed")
        lat_src = lat_src[lat_src["date"] == lat_src["date"].max()]
        lat_src = lat_src[np.isfinite(lat_src["Lat"]) & np.isfinite(lat_src["Long_"])]
        lat_src = lat_src.assign(_w=np.maximum(lat_src["value"], 0) + 1e-9)

        def _wavg_us(g, col):
            return float(np.average(g[col], weights=g["_w"])) if len(g) else np.nan

        lat_map = lat_src.groupby("Province_State").apply(_wavg_us, "Lat", include_groups=False).to_dict()
        lon_map = lat_src.groupby("Province_State").apply(_wavg_us, "Long_", include_groups=False).to_dict()
        pop = load_us("deaths")[["Province_State", "Population"]].drop_duplicates()
        pop_map = pop.set_index("Province_State")["Population"].to_dict()

    dates = np.sort(conf["date"].unique())
    regions = np.sort(conf["region"].unique())
    cw = conf.pivot(index="region", columns="date", values="value").reindex(index=regions, columns=dates).fillna(0.0)
    dw = dead.pivot(index="region", columns="date", values="value").reindex(index=regions, columns=dates).fillna(0.0)

    qc_rows, c_rep, d_rep, c_inc, d_inc = [], {}, {}, {}, {}
    for r in regions:
        cr, ci, revc = repair_monotone(cw.loc[r].to_numpy())
        dr, di, revd = repair_monotone(dw.loc[r].to_numpy())
        c_rep[r], d_rep[r], c_inc[r], d_inc[r] = cr, dr, ci, di
        fin_c, fin_d = cr[-1], dr[-1]
        qc_rows.append(dict(
            region=r, lat=lat_map.get(r, np.nan), lon=lon_map.get(r, np.nan),
            population=(pop_map.get(r, np.nan) if scope != "global" else np.nan),
            final_confirmed=fin_c, final_deaths=fin_d,
            rev_confirmed=revc, rev_deaths=revd,
            rev_pct=100.0 * revc / fin_c if fin_c > 0 else 0.0,
        ))

    qc = pd.DataFrame(qc_rows)
    if min_final_cases:
        qc = qc[qc["final_confirmed"] >= min_final_cases].reset_index(drop=True)
        regions = qc["region"].tolist()

    C = pd.DataFrame([c_rep[r] for r in regions], index=regions, columns=dates)
    D = pd.DataFrame([d_rep[r] for r in regions], index=regions, columns=dates)
    CI = pd.DataFrame([c_inc[r] for r in regions], index=regions, columns=dates)
    DI = pd.DataFrame([d_inc[r] for r in regions], index=regions, columns=dates)

    long = pd.DataFrame({
        "region": np.repeat(regions, len(dates)),
        "date": np.tile(dates, len(regions)),
        "confirmed": C.to_numpy().ravel(),
        "deaths": D.to_numpy().ravel(),
        "new_confirmed": CI.to_numpy().ravel(),
        "new_deaths": DI.to_numpy().ravel(),
    })
    if rec is not None:
        rw = rec.pivot(index="region", columns="date", values="value").reindex(index=regions, columns=dates).fillna(0.0)
        long["recovered"] = rw.to_numpy().ravel()
    long["lat"] = long["region"].map(lat_map)
    long["lon"] = long["region"].map(lon_map)
    long["month"] = long["date"].dt.month
    return long, qc


# ---------------------------------------------------------------- 波次切分
def smooth(series: np.ndarray, win: int = 7) -> np.ndarray:
    return pd.Series(series).rolling(win, center=True, min_periods=1).mean().to_numpy()


def detect_waves(inc_smooth: np.ndarray, dates, up_frac: float = 0.05,
                 peak_frac: float = 0.60, down_frac: float = 0.35,
                 min_len: int = 21, min_peak: float = 5.0):
    """
    相对阈值波次切分。
      - 全局峰值 G = max(平滑新增)
      - 上升段起点: 序列上穿 up_frac*G
      - 平仓/下降段终点: 序列下穿 down_frac*G
      - 峰区: >= peak_frac*G 的连续段
    返回 list of dict(start_idx, end_idx, phase, peak)
    """
    x = np.asarray(inc_smooth, dtype=float)
    G = x.max() if len(x) else 0.0
    waves = []
    if G <= 0:
        return waves
    hi_th, lo_th, pk_th = up_frac * G, down_frac * G, peak_frac * G

    i, n = 0, len(x)
    while i < n:
        if x[i] >= hi_th:
            s = i
            # 前进到低于 lo_th
            j = i
            while j < n and x[j] >= lo_th:
                j += 1
            e = min(j, n - 1)
            if (e - s + 1) >= min_len and x[s:e + 1].max() >= min_peak:
                seg = x[s:e + 1]
                pk = seg.max()
                pk_pos = int(np.argmax(seg)) + s
                # 峰区：>= pk_th 的连续段（围绕 pk_pos）
                a = pk_pos
                while a - 1 >= s and x[a - 1] >= pk_th:
                    a -= 1
                b = pk_pos
                while b + 1 <= e and x[b + 1] >= pk_th:
                    b += 1
                trio = [
                    ("rising", s, max(a - 1, s)),
                    ("peak", a, b),
                    ("declining", min(b + 1, e), e),
                ]
                for phase, ps, pe in trio:
                    if pe < ps:
                        continue
                    waves.append(dict(start=ps, end=pe, phase=phase,
                                      peak=pk, start_date=dates[ps], end_date=dates[pe]))
            i = e + 1
        else:
            i += 1
    return waves
