"""
Step 9: 修正大国坐标 —— JHU 给的是"地理中心"而非"人口/疫情中心"
==================================================================
俄罗斯 (61.5N, 105.3E) = 中西伯利亚, 而 75% 人口在欧洲部分的 55N 附近;
加拿大 (53.9N, -116.6W) = 艾伯塔, 而 70% 人口在 45N 以南的美加边境;
巴西 (-14.2, -51.9) = 内陆高原, 而疫情中心在圣保罗-里约轴线 (-23, -46)。
对"气温"这个自变量而言这是典型的**经典测量误差**, 会向 0 衰减估计量。
修正后重新抓气温并重估。
"""
from __future__ import annotations
import sys, json, os, time
import numpy as np, pandas as pd, urllib.request, urllib.parse

# 人口/疫情中心坐标覆盖表 (lat, lon)
OVERRIDE = {
    "Russia": (55.0, 40.0),          # 欧洲部分人口重心
    "Canada": (45.4, -74.0),         # 圣劳伦斯河-五大湖人口带
    "US": (39.0, -86.0),             # 美国本土人口重心(印第安纳)
    "Brazil": (-21.5, -45.5),        # 圣保罗-里约轴线
    "Argentina": (-34.6, -58.4),     # 布宜诺斯艾利斯
    "Mexico": (19.4, -99.1),         # 墨西哥城
    "Peru": (-12.0, -76.9),          # 利马
    "Turkey": (40.0, 32.0),          # 安卡拉-伊斯坦布尔带
    "Iran": (35.0, 51.5),            # 德黑兰
    "Indonesia": (-7.0, 110.5),      # 爪哇岛
    "South Africa": (-27.0, 27.0),   # 豪登省
    "Chile": (-33.5, -70.6),         # 圣地亚哥
    "Philippines": (14.6, 121.0),    # 马尼拉
    "Japan": (35.7, 139.7),          # 东京
    "India": (22.0, 79.0),           # 人口重心
    "Colombia": (4.7, -74.1),        # 波哥大
}

START, END = "2019-12-01", "2023-04-30"
CACHE = "temp_cache_v2.json"
OUT = "daily_temperature_v2.csv"


def fetch(lats, lons, tries=4):
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{v:.3f}" for v in lats),
        "longitude": ",".join(f"{v:.3f}" for v in lons),
        "start_date": START, "end_date": END,
        "daily": "temperature_2m_mean", "timezone": "UTC"})
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    last = None
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(3 * (a + 1))
    raise last


def centers_for(scope, minc):
    import cw_core as cw
    long, qc = cw.build_panel("region", scope, min_final_cases=minc)
    out = {}
    for _, row in qc.iterrows():
        la, lo = float(row.lat), float(row.lon)
        if scope == "global" and row.region in OVERRIDE:
            la, lo = OVERRIDE[row.region]
        out[f"{scope}::{row.region}"] = (la, lo)
    return out


if __name__ == "__main__":
    need = {}
    for scope, minc in (("global", 200_000), ("us", 100_000)):
        need.update(centers_for(scope, minc))
    print(f"地区数 {len(need)}, 其中 {len(OVERRIDE)} 个大国坐标被覆盖")

    base = json.load(open("temp_cache.json"))
    cache = {}
    todo = []
    for k, (la, lo) in need.items():
        scope, region = k.split("::", 1)
        # 全球样本里被覆盖的国家必须重抓; 其余复用旧缓存
        if k in base and not (scope == "global" and region in OVERRIDE):
            cache[k] = base[k]
        else:
            todo.append(k)
    print(f"复用缓存 {len(cache)}, 需重抓 {len(todo)}")

    B = 6
    for i in range(0, len(todo), B):
        chunk = todo[i:i + B]
        la = [need[k][0] for k in chunk]
        lo = [need[k][1] for k in chunk]
        try:
            res = fetch(la, lo)
            if isinstance(res, dict):
                res = [res]
            for k, r in zip(chunk, res):
                cache[k] = {"time": r["daily"]["time"], "t": r["daily"]["temperature_2m_mean"]}
        except Exception as e:
            print(f"  ! 批次失败, 逐点重试: {e}")
            for k in chunk:
                try:
                    r = fetch([need[k][0]], [need[k][1]])
                    r = r[0] if isinstance(r, list) else r
                    cache[k] = {"time": r["daily"]["time"], "t": r["daily"]["temperature_2m_mean"]}
                except Exception as e2:
                    print(f"    ! {k} 失败 {e2}")
                time.sleep(2)
        print(f"  {min(i+B, len(todo))}/{len(todo)}", flush=True)
        json.dump(cache, open(CACHE, "w"))
        time.sleep(1.2)

    rows = []
    for k, v in cache.items():
        scope, region = k.split("::", 1)
        for d, t in zip(v["time"], v["t"]):
            rows.append((scope, region, d, t))
    df = pd.DataFrame(rows, columns=["scope", "region", "date", "temp_c"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"写出 {OUT}: {len(df):,} 行, 缺失 {df.temp_c.isna().sum()}")

    # 对比修正前后
    old = pd.read_csv("daily_temperature.csv")
    old["date"] = pd.to_datetime(old["date"])
    new = df[df.scope == "global"].rename(columns={"temp_c": "new"})
    cmp = old[old.scope == "global"].merge(new, on=["scope", "region", "date"], how="inner")
    d = cmp.groupby("region").apply(lambda g: pd.Series({
        "old_mean": g.temp_c.mean(), "new_mean": g.new.mean(),
        "old_amp": g.groupby(g.date.dt.month).temp_c.mean().max() - g.groupby(g.date.dt.month).temp_c.mean().min(),
        "new_amp": g.groupby(g.date.dt.month).new.mean().max() - g.groupby(g.date.dt.month).new.mean().min(),
    }), include_groups=False)
    d["dT"] = d.new_mean - d.old_mean
    print("\n坐标修正影响最大的国家 (年均温变化):")
    print(d.reindex(d.dT.abs().sort_values(ascending=False).index).head(16)
          .to_string(float_format=lambda v: f"{v:.2f}"))
