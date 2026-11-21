"""Collecte ERA5 reproductible, en UTC, sans modifier les CSV historiques."""

import argparse
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover", "shortwave_radiation", "wind_speed_10m", "wind_direction_10m",
    "precipitation", "soil_moisture_0_to_7cm",
]


def destination(latitude, longitude, distance_km, bearing):
    angular = distance_km / 6371.0088
    lat, lon, angle = map(math.radians, [latitude, longitude, bearing])
    out_lat = math.asin(math.sin(lat) * math.cos(angular) + math.cos(lat) * math.sin(angular) * math.cos(angle))
    out_lon = lon + math.atan2(math.sin(angle) * math.sin(angular) * math.cos(lat), math.cos(angular) - math.sin(lat) * math.sin(out_lat))
    return math.degrees(out_lat), math.degrees(out_lon)


def study_points(config):
    lat, lon = config["center"]["latitude"], config["center"]["longitude"]
    points = [{"point": "center", "latitude": lat, "longitude": lon, "radius_km": 0}]
    for scale, radius in config["radii_km"].items():
        for name, angle in [("north", 0), ("east", 90), ("south", 180), ("west", 270)]:
            plat, plon = destination(lat, lon, radius, angle)
            points.append({"point": f"{scale}_{name}", "latitude": plat, "longitude": plon, "radius_km": radius})
    return points


def annual_chunks(start, end):
    start, end = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    while start <= end:
        stop = min(end, dt.date(start.year, 12, 31))
        yield start.isoformat(), stop.isoformat()
        start = stop + dt.timedelta(days=1)


def request_chunk(session, params, cache):
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    path = cache / f"{key}.json.gz"
    if path.exists():
        with gzip.open(path, "rt") as handle:
            return json.load(handle), key
    for attempt in range(5):
        try:
            response = session.get(API_URL, params=params, timeout=(15, 90))
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 4:
                    response.raise_for_status()
                delay = min(60, max(5, int(response.headers.get("Retry-After", 5 * (attempt + 1)))))
                print(f"HTTP {response.status_code}, nouvelle tentative dans {delay}s", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                raise ValueError(payload.get("reason", "Réponse API invalide"))
            if payload.get("utc_offset_seconds") != 0 or "hourly" not in payload:
                raise ValueError("La réponse n'est pas une série horaire UTC.")
            # Écriture atomique : une interruption ne doit pas laisser un cache partiel.
            temporary = path.with_suffix(".tmp")
            with gzip.open(temporary, "wt") as handle:
                json.dump(payload, handle, allow_nan=False)
            temporary.replace(path)
            return payload, key
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 4:
                raise
            time.sleep(min(30, 5 * (attempt + 1)))
    raise RuntimeError("Collecte interrompue après plusieurs tentatives.")


def collect(config, output, center_only=False):
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)
    points = study_points(config)
    if center_only:
        points = points[:1]
    frames, records, cells = [], [], {}
    with requests.Session() as session:
        for point in points:
            for start, end in annual_chunks(config["start_date"], config["end_date"]):
                print(f"{point['point']} : {start} → {end}", flush=True)
                params = dict(
                    latitude=point["latitude"], longitude=point["longitude"],
                    start_date=start, end_date=end, hourly=",".join(VARIABLES),
                    models="era5", timezone="UTC", wind_speed_unit="ms",
                    cell_selection="nearest", elevation="nan",
                )
                payload, key = request_chunk(session, params, cache)
                frame = pd.DataFrame(payload["hourly"])
                frame["time"] = pd.to_datetime(frame["time"], utc=True)
                frame["point"] = point["point"]
                frame["lat"], frame["lon"] = payload["latitude"], payload["longitude"]
                if frame.duplicated("time").any():
                    raise ValueError(f"Heures dupliquées : {point['point']}")
                if frame[VARIABLES].isna().any().any():
                    raise ValueError(f"Observations absentes : {point['point']} {start} {end}")
                expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h", tz="UTC")
                if not pd.DatetimeIndex(frame["time"]).equals(expected):
                    raise ValueError(f"Heures manquantes : {point['point']}")
                actual = (payload["latitude"], payload["longitude"])
                if point["point"] in cells and cells[point["point"]] != actual:
                    raise ValueError("La cellule renvoyée change entre deux années.")
                cells[point["point"]] = actual
                frames.append(frame)
                records.append({"point": point["point"], "start": start, "end": end, "cache_sha256": key, "rows": len(frame)})
    if len(set(cells.values())) != len(points):
        raise ValueError("Plusieurs points correspondent à la même cellule ERA5.")
    frame = pd.concat(frames, ignore_index=True).sort_values(["point", "time"])
    name = "center_hourly.csv" if center_only else "spatial_hourly.csv"
    target = output / name
    temporary = target.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(target)
    manifest = {
        "source": API_URL, "model": "era5", "timezone": "UTC",
        "elevation": "nan (no elevation downscaling)",
        "retrieved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": config, "rows": len(frame),
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "points": [{**p, "actual_latitude": cells[p["point"]][0], "actual_longitude": cells[p["point"]][1]} for p in points],
        "chunks": records,
    }
    manifest_path = output / name.replace(".csv", "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"Enregistré : {target} ({len(frame):,} lignes)", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/research_v1")
    parser.add_argument("--center-only", action="store_true")
    args = parser.parse_args()
    collect(json.loads(args.protocol.read_text()), args.output, args.center_only)


if __name__ == "__main__":
    main()
