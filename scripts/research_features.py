"""Variables locales, horaires et spatiales, observées avant 00 h UTC."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_utils import NON_FEATURES, build_supervised
from heatwave_data import build_daily_features

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/research_v1"


def wind_components(speed, direction):
    """Direction météorologique d'origine ; u vers l'est, v vers le nord, m/s."""
    angle = np.deg2rad(direction)
    return -speed * np.sin(angle), -speed * np.cos(angle)


def validate_hourly(frame):
    frame = frame.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    if frame.duplicated(["point", "time"]).any():
        raise ValueError("Heures dupliquées dans une cellule.")
    frame["date"] = frame["time"].dt.tz_localize(None).dt.normalize()
    for point, group in frame.groupby("point"):
        if len(group[["lat", "lon"]].drop_duplicates()) != 1:
            raise ValueError(f"Cellule instable : {point}")
        expected = pd.date_range(group["time"].min(), group["time"].max(), freq="h")
        if not pd.DatetimeIndex(group.sort_values("time")["time"]).equals(expected):
            raise ValueError(f"Heures manquantes : {point}")
    counts = frame.groupby(["point", "date"])["time"].count()
    if not counts.eq(24).all():
        raise ValueError("Journée UTC incomplète.")
    required = ["temperature_2m", "pressure_msl", "relative_humidity_2m", "wind_speed_10m", "wind_direction_10m"]
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("Valeur horaire manquante ou infinie.")
    frame["u"], frame["v"] = wind_components(frame["wind_speed_10m"], frame["wind_direction_10m"])
    return frame.sort_values(["point", "time"])


def calendar_columns(frame):
    target = frame["target_date"]
    month = target.dt.month
    # Fraction de l'année civile : cohérente pendant les années bissextiles.
    start = pd.to_datetime(target.dt.year.astype(str) + "-01-01")
    end = pd.to_datetime((target.dt.year + 1).astype(str) + "-01-01")
    phase = (target - start).dt.days / (end - start).dt.days
    return pd.DataFrame({
        "calendar_month": month,
        "calendar_day_of_year": target.dt.dayofyear,
        "calendar_year": target.dt.year,
        "calendar_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
        "calendar_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
        "calendar_day_sin": np.sin(2 * np.pi * phase),
        "calendar_day_cos": np.cos(2 * np.pi * phase),
        "calendar_winter": month.isin([12, 1, 2]).astype(int),
        "calendar_spring": month.isin([3, 4, 5]).astype(int),
        "calendar_summer": month.isin([6, 7, 8]).astype(int),
        "calendar_autumn": month.isin([9, 10, 11]).astype(int),
    }, index=frame.index)


def local_tables(hourly):
    center = hourly.loc[hourly["point"].eq("center")].copy()
    if center.empty:
        raise ValueError("Cellule center absente.")
    daily = build_daily_features(center)
    daily["date"] = pd.to_datetime(daily["date"])
    supervised = build_supervised(daily, include_history=True)
    base_columns = [c for c in supervised if c not in NON_FEATURES]
    calendar = calendar_columns(supervised)
    supervised = pd.concat([supervised, calendar], axis=1)

    center["hour"] = center["time"].dt.hour
    indexed = center.set_index(["date", "hour"])
    hour_features = pd.DataFrame(index=pd.Index(sorted(center["date"].unique()), name="date"))
    for variable in ["temperature_2m", "pressure_msl", "relative_humidity_2m", "u", "v"]:
        for hour in [6, 12, 18, 23]:
            hour_features[f"hourly_{variable}_{hour:02d}"] = indexed[variable].xs(hour, level="hour")
    for hours in [6, 12]:
        for variable in ["temperature_2m", "pressure_msl", "u", "v"]:
            recent = indexed[variable].xs(23, level="hour")
            past = indexed[variable].xs(23 - hours, level="hour")
            hour_features[f"hourly_{variable}_change_{hours}h"] = recent - past
    frame = supervised.merge(hour_features, on="date", how="left", validate="one_to_one")
    frame["issue_time"] = pd.to_datetime(frame["target_date"], utc=True)
    frame["last_observation_time"] = pd.to_datetime(frame["date"], utc=True) + pd.Timedelta(hours=23)
    if not (frame["last_observation_time"] < frame["issue_time"]).all():
        raise ValueError("Une observation atteint ou dépasse l'heure d'émission.")

    # Cibles horaires du lendemain : elles ne figurent jamais parmi les entrées.
    targets = indexed["temperature_2m"].unstack("hour")
    targets.columns = [f"target_hour_{hour:02d}" for hour in targets.columns]
    targets.index.name = "target_date"
    frame = frame.merge(targets, on="target_date", how="left", validate="one_to_one")
    return frame, base_columns, list(hour_features.columns)


def fit_gradient(values, coordinates_m):
    """Plan local sur les contrastes avec le centre ; sorties par mètre."""
    coordinates_m = np.asarray(coordinates_m, dtype=float)
    if coordinates_m.shape[1] != 2 or np.linalg.matrix_rank(coordinates_m) != 2:
        raise ValueError("Voisinage insuffisant pour un gradient en deux dimensions.")
    return np.asarray(values) @ np.linalg.pinv(coordinates_m).T


def spatial_columns(hourly):
    cells = hourly[["point", "lat", "lon"]].drop_duplicates().set_index("point")
    if len(cells) != 9 or not cells.index.is_unique or cells.duplicated(["lat", "lon"]).any():
        raise ValueError("Il faut neuf cellules distinctes : centre et huit voisins.")
    summary = hourly.groupby(["date", "point"]).agg(
        temperature_mean=("temperature_2m", "mean"),
        pressure_mean=("pressure_msl", "mean"),
    )
    last = hourly.loc[hourly["time"].dt.hour.eq(23)].set_index(["date", "point"])
    for column in ["temperature_2m", "pressure_msl", "u", "v", "relative_humidity_2m"]:
        summary[column + "_23"] = last[column]
    wide = summary.unstack("point")
    center_lat, center_lon = cells.loc["center", ["lat", "lon"]]
    features = pd.DataFrame(index=wide.index)
    groups = {"local_raw": [], "regional_raw": [], "physics": [], "upwind": []}
    for scale in ["local", "regional"]:
        names = [f"{scale}_{d}" for d in ["north", "east", "south", "west"]]
        coords = np.array([
            [6371008.8 * np.cos(np.deg2rad(center_lat)) * np.deg2rad(cells.loc[name, "lon"] - center_lon),
             6371008.8 * np.deg2rad(cells.loc[name, "lat"] - center_lat)]
            for name in names
        ])
        for name in names:
            for variable in summary.columns:
                column = f"neighbor_{name}_{variable}"
                features[column] = wide[(variable, name)]
                groups[f"{scale}_raw"].append(column)
        for variable, short in [("temperature_2m_23", "temperature"), ("pressure_msl_23", "pressure")]:
            contrasts = np.column_stack([wide[(variable, name)] - wide[(variable, "center")] for name in names])
            gradient = fit_gradient(contrasts, coords)
            for axis, j in [("x", 0), ("y", 1)]:
                # K/m pour la température ; hPa/m pour la pression.
                column = f"physics_{scale}_gradient_{short}_{axis}"
                features[column] = gradient[:, j]
                groups["physics"].append(column)
        adv = -(wide[("u_23", "center")] * features[f"physics_{scale}_gradient_temperature_x"]
                + wide[("v_23", "center")] * features[f"physics_{scale}_gradient_temperature_y"])
        column = f"physics_{scale}_thermal_advection_K_per_hour"
        features[column] = adv * 3600  # Vent m/s × gradient K/m → K/h.
        groups["physics"].append(column)

        velocity = np.column_stack([wide[("u_23", "center")], wide[("v_23", "center")]])
        speed = np.linalg.norm(velocity, axis=1)
        unit_positions = coords / np.linalg.norm(coords, axis=1, keepdims=True)
        # Le point amont est opposé au vecteur vitesse (direction d'origine).
        alignment = -velocity @ unit_positions.T / np.maximum(speed[:, None], 1e-12)
        choice = np.argmax(alignment, axis=1)
        rows = np.arange(len(wide))
        temperatures = np.column_stack([wide[("temperature_2m_23", name)] for name in names])
        pressures = np.column_stack([wide[("pressure_msl_23", name)] for name in names])
        calm = speed < 0.5
        for short, values, central in [
            ("temperature_contrast", temperatures, wide[("temperature_2m_23", "center")]),
            ("pressure_contrast", pressures, wide[("pressure_msl_23", "center")]),
        ]:
            column = f"upwind_{scale}_{short}"
            features[column] = np.where(calm, 0, values[rows, choice] - central)
            groups["upwind"].append(column)
        column = f"upwind_{scale}_travel_time_hours"
        features[column] = np.where(calm, 0, np.linalg.norm(coords, axis=1)[choice] / np.maximum(speed, 0.5) / 3600)
        groups["upwind"].append(column)
        column = f"upwind_{scale}_wind_calm"
        features[column] = calm.astype(int)
        groups["upwind"].append(column)
    return features.reset_index(), groups, cells


def load_research_data(spatial=False):
    path = DATA_DIR / ("spatial_hourly.csv" if spatial else "center_hourly.csv")
    if not path.exists():
        raise FileNotFoundError(f"Collecter les données avec scripts/fetch_spatial_data.py : {path}")
    hourly = validate_hourly(pd.read_csv(path))
    frame, daily_columns, hourly_columns = local_tables(hourly)
    groups = {"daily": daily_columns, "hourly": hourly_columns}
    cells = hourly[["point", "lat", "lon"]].drop_duplicates()
    if spatial:
        features, extra_groups, cells = spatial_columns(hourly)
        frame = frame.merge(features, on="date", how="left", validate="one_to_one")
        groups.update(extra_groups)
    frame = frame.sort_values("target_date").reset_index(drop=True)
    all_features = [c for columns in groups.values() for c in columns]
    targets = [f"target_hour_{h:02d}" for h in range(24)]
    if frame[all_features + targets].isna().any().any():
        raise ValueError("Entrées ou cibles incomplètes : pas de suppression silencieuse de jours.")
    if not np.allclose(frame[targets].max(axis=1), frame["tmax_next_day"]):
        raise ValueError("Tmax et maximum des 24 cibles horaires sont incohérents.")
    forbidden = set(NON_FEATURES + targets + ["issue_time", "last_observation_time"])
    if forbidden.intersection(all_features):
        raise ValueError("Une cible ou métadonnée est présente parmi les prédicteurs.")
    return frame, groups, cells, path


def protocol():
    return json.loads((ROOT / "experiments/protocol.json").read_text())
