"""Télécharger Nantes et quatre cellules ERA5 voisines, puis calculer les moyennes par jour."""

from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data/weather_five_points_daily.csv"

# Même période que le dataset de départ.
original_data = pd.read_csv(PROJECT_ROOT / "data/nantes_weather_daily.csv")
START_DATE = original_data["date"].min()
END_DATE = original_data["date"].max()

# Coordonnées des cellules ERA5 : environ 75–85 km entre le centre et chaque voisin.
# La cellule de Nantes est celle déjà utilisée dans les anciens CSV.
POINTS = {
    "nantes": (47.25, -1.50),
    "nord": (48.00, -1.50),
    "sud": (46.50, -1.50),
    "est": (47.25, -0.50),
    "ouest": (47.25, -2.50),
}

VARIABLES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m",
    "pressure_msl", "cloud_cover", "shortwave_radiation",
    "wind_speed_10m", "wind_direction_10m",
    "precipitation", "soil_moisture_0_to_7cm",
]

daily_tables = []
actual_cells = []

for name, (latitude, longitude) in POINTS.items():
    print(f"Téléchargement : {name}", flush=True)

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(VARIABLES),
            "models": "era5",
            "timezone": "UTC",
            "wind_speed_unit": "ms",
            "cell_selection": "nearest",
            "elevation": "nan",  # Garder la température de la cellule sans correction d'altitude.
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()

    hourly = pd.DataFrame(result["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"], utc=True)
    hourly["date"] = hourly["time"].dt.tz_localize(None).dt.normalize()

    # On refuse les trous : chaque journée UTC doit avoir exactement 24 heures.
    expected_hours = pd.date_range(START_DATE, END_DATE + " 23:00", freq="h", tz="UTC")
    if not pd.DatetimeIndex(hourly["time"]).equals(expected_hours):
        raise ValueError(f"Heures manquantes ou dupliquées pour {name}.")
    if hourly[VARIABLES].isna().any().any():
        raise ValueError(f"Valeurs météo manquantes pour {name}.")

    # La direction météo indique D'OÙ vient le vent.
    # u > 0 : vers l'est ; v > 0 : vers le nord. La vitesse est en m/s.
    theta = np.deg2rad(hourly["wind_direction_10m"])
    hourly["u"] = -hourly["wind_speed_10m"] * np.sin(theta)
    hourly["v"] = -hourly["wind_speed_10m"] * np.cos(theta)

    daily = hourly.groupby("date", as_index=False).agg(
        temperature_max=("temperature_2m", "max"),
        temperature_min=("temperature_2m", "min"),
        temperature_mean=("temperature_2m", "mean"),
        relative_humidity_mean=("relative_humidity_2m", "mean"),
        dew_point_mean=("dew_point_2m", "mean"),
        pressure_msl_mean=("pressure_msl", "mean"),
        cloud_cover_mean=("cloud_cover", "mean"),
        wind_speed_mean=("wind_speed_10m", "mean"),
        wind_speed_max=("wind_speed_10m", "max"),
        precipitation_sum=("precipitation", "sum"),
        soil_moisture_mean=("soil_moisture_0_to_7cm", "mean"),
        radiation_hourly_sum=("shortwave_radiation", "sum"),
        u_mean=("u", "mean"),
        v_mean=("v", "mean"),
    )
    # Une moyenne horaire de rayonnement en W/m² devient une énergie en MJ/m².
    daily["shortwave_radiation_sum_MJ_m2"] = daily.pop("radiation_hourly_sum") * 3600 / 1_000_000
    daily["point"] = name
    daily["lat"] = result["latitude"]
    daily["lon"] = result["longitude"]
    actual_cells.append((result["latitude"], result["longitude"]))
    daily_tables.append(daily)
    print(f"  Cellule : {result['latitude']}, {result['longitude']} ; {len(daily)} jours", flush=True)

if len(set(actual_cells)) != 5:
    raise ValueError("Certains points correspondent à la même cellule ERA5.")

data = pd.concat(daily_tables, ignore_index=True)
OUTPUT_PATH.parent.mkdir(exist_ok=True)
data.to_csv(OUTPUT_PATH, index=False)
print(f"Fichier enregistré : {OUTPUT_PATH} ({len(data)} lignes)")
