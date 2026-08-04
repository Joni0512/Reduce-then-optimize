"""
Descriptive statistics for the NYC master file (data/requests/requests.csv) -
the pooled, no-date file used for the dense2000/dense900-style synthetic
stress-test slices. Distinct dataset from data_for_experiments_NYC/requests_*.csv
(see nyc_descriptive_stats.py) - confirmed zero row overlap, different distance
filtering profile (see docs/NYC_Dataset.md "Source & Size" correction).

Usage:
    python3 scripts/analysis/nyc_master_descriptive_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")
MASTER_CSV = RH_ROOT / "data" / "requests" / "requests.csv"
NODES_CSV = RH_ROOT / "data" / "map" / "nodes.csv"
COLS = ["request_id", "pickup_node", "pickup_lon", "pickup_lat",
        "dropoff_node", "dropoff_lon", "dropoff_lat", "time"]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def main():
    df = pd.read_csv(MASTER_CSV, header=None, names=COLS)
    df["t"] = pd.to_timedelta(df["time"])
    df["hour"] = (df["t"].dt.total_seconds() // 3600).astype(int)
    df["min_of_day"] = (df["t"].dt.total_seconds() // 60).astype(int)

    print("=" * 70)
    print("VARIABLES / COLUMNS")
    print("=" * 70)
    print(f"Columns: {COLS}")
    print("Note: node_id ALREADY resolved (unlike requests_{1,10,20}.csv) - ")
    print("no date column - time-of-day only, pooled across an unconfirmed")
    print("number of underlying days (see docs/NYC_Dataset.md).\n")

    print("=" * 70)
    print("SIZE")
    print("=" * 70)
    print(f"Total requests: {len(df)}")
    print(f"Time range: {df['time'].min()} .. {df['time'].max()}\n")

    print("=" * 70)
    print("NODE ID COVERAGE")
    print("=" * 70)
    print(f"Unique pickup nodes: {df['pickup_node'].nunique()}")
    print(f"Unique dropoff nodes: {df['dropoff_node'].nunique()}")
    print(f"Pickup node id range: {df['pickup_node'].min()} .. {df['pickup_node'].max()}")
    print(f"Dropoff node id range: {df['dropoff_node'].min()} .. {df['dropoff_node'].max()}")
    if NODES_CSV.exists():
        nodes = pd.read_csv(NODES_CSV, header=None, names=["node_id", "lat", "lon"])
        print(f"data/map/nodes.csv total nodes: {len(nodes)}")
        unknown_pickup = (~df["pickup_node"].isin(nodes["node_id"])).sum()
        unknown_dropoff = (~df["dropoff_node"].isin(nodes["node_id"])).sum()
        print(f"pickup_node values NOT found in nodes.csv: {unknown_pickup}")
        print(f"dropoff_node values NOT found in nodes.csv: {unknown_dropoff}")
    print()

    print("=" * 70)
    print("HOURLY REQUEST DENSITY (whole file, no date -> pooled across all")
    print("underlying days, so this is total volume per hour-of-day, not a")
    print("single day's pattern)")
    print("=" * 70)
    by_hour = df.groupby("hour").size()
    for h in range(24):
        c = by_hour.get(h, 0)
        bar = "#" * int(c / by_hour.max() * 50)
        flag = "  <-- peak" if c == by_hour.max() else ("  <-- trough" if c == by_hour.min() else "")
        print(f"{h:02d}:00  {c:>6}  {bar}{flag}")

    print("\n" + "=" * 70)
    print("DENSEST 30-MINUTE WINDOWS (top 10, by request count)")
    print("=" * 70)
    df["bucket30"] = (df["min_of_day"] // 30 * 30).astype(int)
    by_bucket = df.groupby("bucket30").size().sort_values(ascending=False)
    for b, c in by_bucket.head(10).items():
        hh, mm = divmod(b, 60)
        print(f"{hh:02d}:{mm:02d}-{(hh if mm==0 else hh):02d}:{(mm+30)%60:02d}  {c:>6} requests")

    print("\nSPARSEST 30-minute windows (bottom 5):")
    for b, c in by_bucket.tail(5).items():
        hh, mm = divmod(b, 60)
        print(f"{hh:02d}:{mm:02d}  {c:>6} requests")

    print("\n" + "=" * 70)
    print("TRIP DISTANCE (haversine, pickup->dropoff)")
    print("=" * 70)
    dist = haversine_m(df["pickup_lat"], df["pickup_lon"], df["dropoff_lat"], df["dropoff_lon"])
    print(f"Mean: {dist.mean():.0f} m   Median: {dist.median():.0f} m   "
          f"Min: {dist.min():.0f} m   Max: {dist.max():.0f} m   Std: {dist.std():.0f} m")
    for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
        print(f"  p{int(q*100):>2}: {dist.quantile(q):>7.0f} m")
    print(f"Fraction under 1km: {(dist<1000).mean()*100:.1f}%")
    print(f"Fraction under 5km: {(dist<5000).mean()*100:.1f}%")
    print(f"Fraction over 10km: {(dist>10000).mean()*100:.1f}%")

    print("\n" + "=" * 70)
    print("SPATIAL SPREAD")
    print("=" * 70)
    print(f"Pickup lat range:  {df['pickup_lat'].min():.4f} .. {df['pickup_lat'].max():.4f}")
    print(f"Pickup lon range:  {df['pickup_lon'].min():.4f} .. {df['pickup_lon'].max():.4f}")
    print(f"Dropoff lat range: {df['dropoff_lat'].min():.4f} .. {df['dropoff_lat'].max():.4f}")
    print(f"Dropoff lon range: {df['dropoff_lon'].min():.4f} .. {df['dropoff_lon'].max():.4f}")


if __name__ == "__main__":
    main()
