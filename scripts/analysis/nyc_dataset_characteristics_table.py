"""
Builds one comparable characteristics table across all four NYC request files:
requests_1.csv / requests_10.csv / requests_20.csv (dated, data_for_experiments_NYC)
and requests.csv (master, pooled, no date - data/requests).

Writes markdown + CSV output into external_repos/RollingHorizon/.

Usage:
    python3 scripts/analysis/nyc_dataset_characteristics_table.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

RH_ROOT = Path("/Users/joni/Desktop/Masterarbeit/external_repos/RollingHorizon")
SAMPLE_DIR = RH_ROOT / "data_for_experiments_NYC"
MASTER_CSV = RH_ROOT / "data" / "requests" / "requests.csv"
OUT_MD = RH_ROOT / "nyc_dataset_characteristics.md"
OUT_CSV = RH_ROOT / "nyc_dataset_characteristics.csv"

DATED_COLS = ["request_id", "pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat", "time", "date"]
MASTER_COLS = ["request_id", "pickup_node", "pickup_lon", "pickup_lat",
               "dropoff_node", "dropoff_lon", "dropoff_lat", "time"]


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_dated(name):
    df = pd.read_csv(SAMPLE_DIR / f"requests_{name}.csv", header=0, names=DATED_COLS)
    df["dt"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M:%S")
    df["date_only"] = df["dt"].dt.date
    df["hour"] = df["dt"].dt.hour
    df["min_of_day"] = df["dt"].dt.hour * 60 + df["dt"].dt.minute
    df["has_date"] = True
    return df


def load_master():
    df = pd.read_csv(MASTER_CSV, header=None, names=MASTER_COLS)
    t = pd.to_timedelta(df["time"])
    df["hour"] = (t.dt.total_seconds() // 3600).astype(int)
    df["min_of_day"] = (t.dt.total_seconds() // 60).astype(int)
    df["has_date"] = False
    return df


def characterize(name, df):
    dist = haversine_m(df["pickup_lat"], df["pickup_lon"], df["dropoff_lat"], df["dropoff_lon"])
    by_hour = df.groupby("hour").size()
    df = df.copy()
    df["bucket30"] = (df["min_of_day"] // 30 * 30).astype(int)
    by_bucket = df.groupby("bucket30").size().sort_values(ascending=False)
    busiest_b = by_bucket.index[0]
    sparsest_b = by_bucket.index[-1]

    def fmt_bucket(b):
        hh, mm = divmod(int(b), 60)
        return f"{hh:02d}:{mm:02d}"

    row = {
        "dataset": name,
        "n_requests": len(df),
        "has_real_date": df["has_date"].iloc[0] if "has_date" in df else None,
        "unique_days": df["date_only"].nunique() if "date_only" in df.columns else "N/A (no date col)",
        "date_range": (f"{df['date_only'].min()}..{df['date_only'].max()}"
                        if "date_only" in df.columns else "N/A (pooled, unconfirmed)"),
        "peak_hour": f"{by_hour.idxmax():02d}:00 ({by_hour.max()})",
        "trough_hour": f"{by_hour.idxmin():02d}:00 ({by_hour.min()})",
        "busiest_30min": f"{fmt_bucket(busiest_b)} ({by_bucket.iloc[0]})",
        "sparsest_30min": f"{fmt_bucket(sparsest_b)} ({by_bucket.iloc[-1]})",
        "dist_mean_m": round(dist.mean()),
        "dist_median_m": round(dist.median()),
        "dist_min_m": round(dist.min()),
        "dist_max_m": round(dist.max()),
        "pct_under_5km": round((dist < 5000).mean() * 100, 1),
        "pickup_lat_range": f"{df['pickup_lat'].min():.4f}..{df['pickup_lat'].max():.4f}",
        "pickup_lon_range": f"{df['pickup_lon'].min():.4f}..{df['pickup_lon'].max():.4f}",
    }
    if "date_only" in df.columns:
        per_day = df.groupby("date_only").size()
        row["requests_per_day_mean"] = round(per_day.mean())
        row["requests_per_day_min"] = f"{per_day.min()} ({per_day.idxmin()})"
        row["requests_per_day_max"] = f"{per_day.max()} ({per_day.idxmax()})"
        low_days = per_day[per_day < per_day.mean() * 0.5]
        row["day_outliers"] = "; ".join(f"{d} ({c})" for d, c in low_days.items()) or "none < 50% of mean"
    else:
        row["requests_per_day_mean"] = "N/A"
        row["requests_per_day_min"] = "N/A"
        row["requests_per_day_max"] = "N/A"
        row["day_outliers"] = "N/A (no date col, cannot separate days)"
    return row


def main():
    frames = {
        "1%": load_dated("1"),
        "10%": load_dated("10"),
        "20%": load_dated("20"),
        "master (full, pooled)": load_master(),
    }
    rows = [characterize(name, df) for name, df in frames.items()]
    table = pd.DataFrame(rows).set_index("dataset").T

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_CSV)

    def to_markdown_manual(df):
        cols = list(df.columns)
        header = "| " + " | ".join(["characteristic"] + cols) + " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1)) + "|"
        lines = [header, sep]
        for idx, row in df.iterrows():
            vals = [str(row[c]) for c in cols]
            lines.append("| " + " | ".join([str(idx)] + vals) + " |")
        return "\n".join(lines)

    md_table = to_markdown_manual(table)

    with open(OUT_MD, "w") as f:
        f.write("# NYC Request Datasets — Characteristics Comparison\n\n")
        f.write(md_table)
        f.write("\n")

    print(md_table)
    print(f"\nWritten to:\n  {OUT_MD}\n  {OUT_CSV}")


if __name__ == "__main__":
    main()
