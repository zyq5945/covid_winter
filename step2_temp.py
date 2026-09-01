"""Step 2: 抓取每个地区代表点的逐日实测气温 (Open-Meteo / ERA5 再分析)
代表点 = 病例加权经纬度中心。用于把"寒冷"从纬度代理升级为实测气温。"""
import sys, io, json, time, os
import numpy as np, pandas as pd, urllib.request, urllib.parse

START, END = "2019-12-01", "2023-04-30"
OUT = "daily_temperature.csv"
CACHE = "temp_cache.json"


def fetch_batch(lats, lons):
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{v:.3f}" for v in lats),
        "longitude": ",".join(f"{v:.3f}" for v in lons),
        "start_date": START, "end_date": END,
        "daily": "temperature_2m_mean", "timezone": "UTC",
    })
    url = "https://archive-api.open-meteo.com/v1/archive?" + q
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last


def fetch_one(lat, lon):
    try:
        r = fetch_batch([lat], [lon])
        return r[0] if isinstance(r, list) else r
    except Exception as e:
        print(f"    ! 单点重试也失败 ({lat:.2f},{lon:.2f}): {e}")
        return None


def main(scopes=(("global", 200_000), ("us", 100_000))):
    import cw_core as cw
    centers = {}
    for scope, minc in scopes:
        long, qc = cw.build_panel("region", scope, min_final_cases=minc)
        for _, row in qc.iterrows():
            la, lo = float(row.lat), float(row.lon)
            if not (np.isfinite(la) and np.isfinite(lo)):
                print(f"  ! 跳过无效坐标: {scope}::{row.region}")
                continue
            centers[f"{scope}::{row.region}"] = (la, lo)
    print(f"待抓取地区数: {len(centers)}")

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    keys = [k for k in centers if k not in cache]
    print(f"缓存命中 {len(centers)-len(keys)}, 待请求 {len(keys)}")

    B = 8
    for i in range(0, len(keys), B):
        chunk = keys[i:i + B]
        lats = [centers[k][0] for k in chunk]
        lons = [centers[k][1] for k in chunk]
        try:
            res = fetch_batch(lats, lons)
            if isinstance(res, dict):
                res = [res]
            for k, r in zip(chunk, res):
                cache[k] = {"time": r["daily"]["time"], "t": r["daily"]["temperature_2m_mean"]}
        except Exception as e:
            print(f"  ! 批次失败({len(chunk)})，逐点重试: {e}")
            for k in chunk:
                la, lo = centers[k]
                r = fetch_one(la, lo)
                if r is not None:
                    cache[k] = {"time": r["daily"]["time"], "t": r["daily"]["temperature_2m_mean"]}
        print(f"  {min(i+B, len(keys))}/{len(keys)}", flush=True)
        json.dump(cache, open(CACHE, "w"))

    # 展开为长表
    rows = []
    for k, v in cache.items():
        scope, region = k.split("::", 1)
        for d, t in zip(v["time"], v["t"]):
            rows.append((scope, region, d, t))
    df = pd.DataFrame(rows, columns=["scope", "region", "date", "temp_c"])
    df["date"] = pd.to_datetime(df["date"])
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"写出 {OUT}: {len(df):,} 行, 缺失 {df.temp_c.isna().sum():,}")


if __name__ == "__main__":
    main()
