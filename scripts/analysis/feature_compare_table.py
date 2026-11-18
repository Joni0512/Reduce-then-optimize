"""
2026-11-17: explained side-by-side table of the actor features (feat_builder.py v1, 85 features) on SF vs Li&Lim
(user: "die Features auch erklaeren in der Tabelle"). Input: outputs/feature_check/feature_stats_{lilim,sf}.csv from
scripts/analysis/feature_stats_sf_vs_lilim.py (random-init actor, one eval episode each; see that docstring).
Output: figures_export/SF/actor_features_sf_vs_lilim_table.csv; prints a markdown table (the 49 fr_grid cells pooled).
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs" / "feature_check"
OUT = ROOT / "figures_export" / "SF" / "actor_features_sf_vs_lilim_table.csv"

DESC = {
    "action_reject_flag": "1 = Zeile ist die 'Request ablehnen'-Aktion, 0 = echter Trip",
    "cr_pickup_slack": "Zeit bis zur spaetesten Pickup-Zeit des dringendsten Requests im Trip, in batch_interval-Einheiten",
    "s_norm_time": "Zeitpunkt im Tag, 0 = erster Request, 1 = letzter Request",
    "s_avg_remaining_am_cap": "mittlere freie Sitzplatz-Kapazitaet der Flotte (Anteil)",
    "s_avg_remaining_wc_cap": "mittlere freie Rollstuhl-Kapazitaet der Flotte (Anteil)",
    "s_total_vehicles": "Anzahl Fahrzeuge (Rohzahl, pro Instanz konstant)",
    "s_avg_remaining_interval_boarded_time": "mittlere restliche Fahrzeit mit Fahrgaesten an Bord / batch_interval",
    "s_avg_remaining_step_boarded_time": "mittlere restliche Fahrzeit mit Fahrgaesten an Bord / step_size",
    "t_num_requests_in_trip": "Anzahl neuer Requests im Kandidaten-Trip (1 oder 2)",
    "t_norm_num_requests_by_vehicle_max_capacity": "Requests im Trip / Fahrzeug-Maximalkapazitaet",
    "v_norm_lat_next_position": "Nord-Sued-Position des naechsten Fahrzeug-Knotens, 0-1 im Bediengebiet",
    "v_norm_lon_next_position": "Ost-West-Position des naechsten Fahrzeug-Knotens, 0-1 im Bediengebiet",
    "v_norm_remaining_operating_period": "Anteil der verbleibenden Dienstzeit des Fahrzeugs",
    "v_norm_vehicle_count_in_proximity": "Anteil anderer Fahrzeuge im Umkreis von 1000 m (geodesic)",
    "v_avg_vehicle_distance": "mittlerer Abstand zu allen anderen Fahrzeugen / (0.5 * max_distance)",
    "v_norm_step_remaining_boarded_time": "restliche Fahrzeit mit Fahrgaesten dieses Fahrzeugs / step_size",
    "v_norm_interval_remaining_boarded_time": "restliche Fahrzeit mit Fahrgaesten dieses Fahrzeugs / batch_interval",
    "v_norm_remaining_am_cap": "freie Sitzplatz-Kapazitaet des Fahrzeugs (Anteil)",
    "v_norm_remaining_wc_cap": "freie Rollstuhl-Kapazitaet des Fahrzeugs (Anteil)",
    "v_am_cap": "Sitzplatz-Kapazitaet des Fahrzeugs (Rohzahl)",
    "v_wc_cap": "Rollstuhl-Kapazitaet des Fahrzeugs (Rohzahl)",
    "tc_sequence_len": "Anzahl Stopps in der geplanten Route nach Einfuegen",
    "tc_num_trips": "Anzahl Trips/Requests im Plan des Fahrzeugs",
    "tc_travel_time_to_first_pickup": "Fahrzeit bis zum ersten Pickup [Sekunden, roh]",
    "tc_norm_travel_time_to_first_pickup": "Fahrzeit bis zum ersten Pickup / max_distance (Sekunden / METER)",
    "tc_total_direct_travel_time": "Summe der Direktfahrzeiten der Requests [Sekunden, roh]",
    "tc_actual_travel_time": "tatsaechliche Fahrzeit der Route [Sekunden, roh]",
    "tc_norm_batch_actual_travel_time": "tatsaechliche Fahrzeit der Route / batch_interval",
    "tc_total_dwell_time": "Summe der Haltezeiten an Stopps [Sekunden, roh]",
    "tc_dwell_time_ratio": "Haltezeit / tatsaechliche Fahrzeit",
    "tc_detour_time": "Umwegzeit durch Teilen [Sekunden, roh]",
    "tc_norm_detour_time": "Umwegzeit / max_distance (Sekunden / METER)",
    "tc_norm_idling_time": "Wartezeit vor fruehestem Pickup / batch_interval",
    "tc_sharing_efficiency_factor": "(tatsaechliche - direkte Fahrzeit) / direkte Fahrzeit",
    "tc_total_am_demand": "Sitzplatzbedarf aller Trips im Plan",
    "tc_total_wc_demand": "Rollstuhlbedarf aller Trips im Plan",
    "fr_grid": "49 Zellen (7x7-Raster): Pickup-Zelle(n) der 1-2 Requests DIESES Kandidaten, gewichtet 1/0.5/0.25/0.125 je batch_interval bis Pickup (2026-11-17: build_from_components uebergibt trip_requests, NICHT alle kommenden Requests)",
}


def load(tag):
    with open(SRC / f"feature_stats_{tag}.csv") as fh:
        return {r["feature"]: r for r in csv.DictReader(fh)}


def f(x):
    x = float(x)
    if x == 0:
        return "0"
    return f"{x:.3g}" if abs(x) < 1e4 else f"{x:.2e}"


def main():
    L, S = load("lilim"), load("sf")
    rows = []
    for feat in sorted(L):
        key = "fr_grid" if feat.startswith("fr_grid") else feat
        l, s = L[feat], S.get(feat, {})
        rows.append({"feature": feat, "erklaerung": DESC.get(key, "?"),
                     **{f"lilim_{k}": l[k] for k in ("min", "median", "p99", "max", "nan", "n_unique")},
                     **{f"sf_{k}": s.get(k) for k in ("min", "median", "p99", "max", "nan", "n_unique")}})
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("| Feature | Erklaerung | Li&Lim median / p99 / max | SF median / p99 / max | SF/Li&Lim p99 | konstant? |")
    print("|---|---|---|---|---|---|")
    grid = [r for r in rows if r["feature"].startswith("fr_grid")]
    for r in [r for r in rows if not r["feature"].startswith("fr_grid")]:
        ratio = float(r["sf_p99"]) / float(r["lilim_p99"]) if float(r["lilim_p99"]) else float("nan")
        const = ", ".join(t for t, n in (("Li&Lim", r["lilim_n_unique"]), ("SF", r["sf_n_unique"])) if int(float(n)) <= 1)
        print(f"| `{r['feature']}` | {r['erklaerung']} | {f(r['lilim_median'])} / {f(r['lilim_p99'])} / {f(r['lilim_max'])} | "
              f"{f(r['sf_median'])} / {f(r['sf_p99'])} / {f(r['sf_max'])} | {ratio:.3g} | {const or '-'} |")
    gl = max(float(r["lilim_max"]) for r in grid); gs = max(float(r["sf_max"]) for r in grid)
    gpl = max(float(r["lilim_p99"]) for r in grid); gps = max(float(r["sf_p99"]) for r in grid)
    cl = sum(int(float(r["lilim_n_unique"])) <= 1 for r in grid); cs = sum(int(float(r["sf_n_unique"])) <= 1 for r in grid)
    print(f"| `fr_grid_*` (49) | {DESC['fr_grid']} | max p99 {f(gpl)} / max {f(gl)} | max p99 {f(gps)} / max {f(gs)} | "
          f"{gps / gpl:.3g} | konstant: Li&Lim {cl}/49, SF {cs}/49 |")
    nan = [r["feature"] for r in rows if int(float(r["lilim_nan"])) or int(float(r["sf_nan"] or 0))]
    print("NaN in:", nan or "keinem Feature")


if __name__ == "__main__":
    main()
