"""
Descriptive statistics for the NYC percentage-sample datasets
(data_for_experiments_NYC/requests_{1,10,20}.csv) - the only NYC files with a
real `date` column (the pooled data/requests/requests.csv master has none).

Usage:
    python3 scripts/analysis/nyc_descriptive_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")
SAMPLE_DIR = RH_ROOT / "data_for_experiments_NYC"
COLS = ["request_id", "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat", "time", "date"]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load(name):
    df = pd.read_csv(SAMPLE_DIR / f"requests_{name}.csv", header=0, names=COLS)
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M:%S")
    df["date_only"] = df["dt"].dt.date
    df["weekday"] = df["dt"].dt.day_name()
    df["hour"] = df["dt"].dt.hour
    return df


def main():
    samples = {name: load(name) for name in ["1", "10", "20"]}

    print("=" * 70)
    print("VARIABLES / COLUMNS")
    print("=" * 70)
    print(f"Raw columns: {COLS}")
    print("Note: no node_id in these files (needs nearest-node snapping before")
    print("use in the C++/Python solvers). No explicit pickup/dropoff time window -")
    print("only a single request timestamp; windows are constructed later during")
    print("manifest-building (MAX_WAITING/MAX_DETOUR assumptions).\n")

    print("=" * 70)
    print("SIZE & NESTING")
    print("=" * 70)
    for name, df in samples.items():
        print(f"requests_{name}.csv: {len(df)} rows")
    r1, r10, r20 = samples["1"], samples["10"], samples["20"]
    key = ["pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat", "time", "date"]
    m1 = r1.merge(r10, on=key, how="left", indicator=True)
    m2 = r10.merge(r20, on=key, how="left", indicator=True)
    print(f"1% rows found in 10% (exact match): {(m1['_merge']=='both').sum()}/{len(r1)}")
    print(f"10% rows found in 20% (exact match): {(m2['_merge']=='both').sum()}/{len(r10)}")
    print("-> samples are NESTED (1% subset of 10% subset of 20%), not independent draws.\n")

    print("=" * 70)
    print("DATE RANGE & PER-DAY COUNTS (20% sample)")
    print("=" * 70)
    df20 = samples["20"]
    per_day = df20.groupby("date_only").size().sort_index()
    print(f"Unique days: {df20['date_only'].nunique()}")
    print(f"Date range: {per_day.index.min()} .. {per_day.index.max()}\n")
    print(f"{'Date':<12} {'Weekday':<10} {'Requests':>8}")
    weekday_map = {d: pd.Timestamp(d).day_name() for d in per_day.index}
    for d, c in per_day.items():
        flag = ""
        if c < 1200:
            flag = "  <-- anomaly (holiday/weather?)"
        print(f"{str(d):<12} {weekday_map[d]:<10} {c:>8}{flag}")

    print(f"\nMean/day: {per_day.mean():.0f}  Median/day: {per_day.median():.0f}  "
          f"Min: {per_day.min()} ({weekday_map[per_day.idxmin()]})  "
          f"Max: {per_day.max()} ({weekday_map[per_day.idxmax()]})")

    by_weekday = df20.groupby("weekday").size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    )
    print("\nTotal requests by weekday (summed across all days in that weekday, 20% sample):")
    for wd, c in by_weekday.items():
        print(f"  {wd:<10} {c:>7}")

    print("\n" + "=" * 70)
    print("HOURLY DEMAND PATTERN (20% sample, aggregated across all 31 days)")
    print("=" * 70)
    by_hour = df20.groupby("hour").size()
    max_hour = by_hour.idxmax()
    min_hour = by_hour.idxmin()
    for h in range(24):
        c = by_hour.get(h, 0)
        bar = "#" * int(c / by_hour.max() * 50)
        flag = "  <-- peak" if h == max_hour else ("  <-- trough" if h == min_hour else "")
        print(f"{h:02d}:00  {c:>5}  {bar}{flag}")

    print("\n" + "=" * 70)
    print("TRIP DISTANCE (haversine, pickup->dropoff), 20% sample")
    print("=" * 70)
    dist = haversine_m(df20["pickup_lat"], df20["pickup_lon"], df20["dropoff_lat"], df20["dropoff_lon"])
    print(f"Mean: {dist.mean():.0f} m   Median: {dist.median():.0f} m   "
          f"Min: {dist.min():.0f} m   Max: {dist.max():.0f} m   Std: {dist.std():.0f} m")
    print(f"Percentiles: 10%={dist.quantile(0.1):.0f}m  25%={dist.quantile(0.25):.0f}m  "
          f"75%={dist.quantile(0.75):.0f}m  90%={dist.quantile(0.9):.0f}m")

    print("\n" + "=" * 70)
    print("SPATIAL SPREAD (20% sample)")
    print("=" * 70)
    print(f"Pickup lat range:  {df20['pickup_lat'].min():.4f} .. {df20['pickup_lat'].max():.4f}")
    print(f"Pickup lon range:  {df20['pickup_lon'].min():.4f} .. {df20['pickup_lon'].max():.4f}")
    print(f"Dropoff lat range: {df20['dropoff_lat'].min():.4f} .. {df20['dropoff_lat'].max():.4f}")
    print(f"Dropoff lon range: {df20['dropoff_lon'].min():.4f} .. {df20['dropoff_lon'].max():.4f}")


if __name__ == "__main__":
    main()
