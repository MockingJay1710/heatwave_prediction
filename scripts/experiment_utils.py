"""Préparation commune aux expériences de prévision de Tmax à J+1."""

import numpy as np
import pandas as pd


HISTORY_FEATURES = [
    "tmax_lag1", "tmax_lag2", "delta_pressure", "delta_humidity",
]
NON_FEATURES = ["date", "target_date", "lat", "lon", "tmax_next_day"]


def prepare_daily_data(data):
    """Valide une seule cellule météo et conserve les unités physiques."""
    frame = data.copy()
    required = {
        "date", "lat", "lon", "temperature_max", "pressure_msl_mean",
        "relative_humidity_mean",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame[["date", "lat", "lon"]].isna().any().any():
        raise ValueError("Date ou coordonnées manquantes.")
    if not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("Le dataset doit contenir des dates journalières à minuit.")
    if len(frame[["lat", "lon"]].drop_duplicates()) != 1:
        raise ValueError("Cette expérience attend une seule cellule météo (Nantes).")
    if frame.duplicated(["date", "lat", "lon"]).any():
        raise ValueError("Doublons de date et de localisation : corriger la source.")
    frame = frame.sort_values("date").reset_index(drop=True)
    weather = frame.drop(columns=["date", "lat", "lon"])
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in weather.dtypes):
        raise ValueError("Les observations météo doivent être numériques.")
    if np.isinf(weather.to_numpy(dtype=float)).any():
        raise ValueError("Valeur météo infinie.")
    return frame


def build_supervised(data, include_history=False):
    """Jointure sur J+1 exact ; retards sur J-1/J-2, même en cas de trou."""
    frame = prepare_daily_data(data)
    daily = frame.set_index("date")
    frame["target_date"] = frame["date"] + pd.Timedelta(days=1)
    frame["tmax_next_day"] = frame["target_date"].map(daily["temperature_max"])
    if include_history:
        for lag in (1, 2):
            frame[f"tmax_lag{lag}"] = (
                frame["date"] - pd.Timedelta(days=lag)
            ).map(daily["temperature_max"])
        previous_dates = frame["date"] - pd.Timedelta(days=1)
        for name, column in [
            ("delta_pressure", "pressure_msl_mean"),
            ("delta_humidity", "relative_humidity_mean"),
        ]:
            frame[name] = frame[column] - previous_dates.map(daily[column])
    # Même population complète pour tous les modèles ; aucune imputation future.
    columns = [c for c in frame.columns if c not in ["lat", "lon"]]
    return frame.dropna(subset=columns).reset_index(drop=True)


def chronological_masks(frame, cutoff):
    """N'utilise à l'entraînement que les cibles strictement avant la coupure."""
    cutoff = pd.Timestamp(cutoff)
    train = frame["target_date"] < cutoff
    test = frame["date"] >= cutoff
    if not train.any() or not test.any():
        raise ValueError("Périodes d'entraînement ou d'évaluation vides.")
    assert frame.loc[train, "target_date"].max() < frame.loc[test, "date"].min()
    return train, test


def add_calendar_features(frame):
    """Calendrier de la date cible, connue à l'avance, sans statistique apprise."""
    frame = frame.copy()
    month = frame["target_date"].dt.month
    # Décembre et janvier restent proches dans la représentation cyclique.
    frame["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    frame["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
    seasons = {
        "winter": [12, 1, 2], "spring": [3, 4, 5],
        "summer": [6, 7, 8], "autumn": [9, 10, 11],
    }
    for name, months in seasons.items():
        frame[f"season_{name}"] = month.isin(months).astype(int)
    return frame
