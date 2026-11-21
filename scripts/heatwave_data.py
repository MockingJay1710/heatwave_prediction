import datetime
from pathlib import Path
import math
import requests
import time
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

START_DATE = "2024-01-01"

# ERA5 a environ 5 jours de délai.
# On prend 6 jours pour être tranquille vis-à-vis de l'heure de mise à jour.
END_DATE = (
    datetime.date.today() - datetime.timedelta(days=6)
).strftime("%Y-%m-%d")

API_URL = "https://archive-api.open-meteo.com/v1/archive"
API_TIMEOUT = 180

MAX_RETRIES = 5


# Point cible : centre de Nantes
NANTES = (47.2184, -1.5536)


# Variables choisies pour notre première expérimentation
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "pressure_msl",
    "cloud_cover",
    "shortwave_radiation",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "soil_moisture_0_to_7cm",
]


# ============================================================
# FETCH D'UN BATCH DE POINTS
# ============================================================

def fetch_batch(points):
    """
    Récupère l'historique horaire de plusieurs points
    en un seul appel à l'API Open-Meteo.

    IMPORTANT :
    lat/lon enregistrés = coordonnées de la cellule ERA5
    réellement renvoyée par Open-Meteo.

    Retour :
        DataFrame contenant une ligne par heure et par point.

    Retourne None si toutes les tentatives échouent.
    """

    params = {
        "latitude": ",".join(str(p[0]) for p in points),
        "longitude": ",".join(str(p[1]) for p in points),

        "start_date": START_DATE,
        "end_date": END_DATE,

        "hourly": ",".join(HOURLY_VARIABLES),

        "timezone": "Europe/Paris",

        "wind_speed_unit": "ms",

        # Même modèle partout et dans le temps
        "models": "era5",

        # On demande réellement la cellule la plus proche
        # plutôt qu'une cellule terrestre optimisée.
        "cell_selection": "nearest",
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            print(
                f"Appel API "
                f"(tentative {attempt}/{MAX_RETRIES})..."
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=API_TIMEOUT,
            )

            # Rate limit
            if response.status_code == 429:
                wait = 20 * attempt

                print(
                    f"Rate limit atteint. "
                    f"Nouvelle tentative dans {wait}s..."
                )

                time.sleep(wait)
                continue

            response.raise_for_status()

            data = response.json()

            # Open-Meteo :
            # - 1 point  -> dict
            # - N points -> list[dict]
            if isinstance(data, dict):
                data = [data]

            frames = []

            for cell in data:

                hourly = cell["hourly"]

                frame = pd.DataFrame({
                    "time": hourly["time"],

                    # Coordonnées de la vraie cellule ERA5 utilisée
                    "lat": cell["latitude"],
                    "lon": cell["longitude"],

                    "temperature_2m":
                        hourly["temperature_2m"],

                    "relative_humidity_2m":
                        hourly["relative_humidity_2m"],

                    "dew_point_2m":
                        hourly["dew_point_2m"],

                    "pressure_msl":
                        hourly["pressure_msl"],

                    "cloud_cover":
                        hourly["cloud_cover"],

                    "shortwave_radiation":
                        hourly["shortwave_radiation"],

                    "wind_speed_10m":
                        hourly["wind_speed_10m"],

                    "wind_direction_10m":
                        hourly["wind_direction_10m"],

                    "precipitation":
                        hourly["precipitation"],

                    "soil_moisture_0_to_7cm":
                        hourly["soil_moisture_0_to_7cm"],
                })

                frames.append(frame)

            result = pd.concat(
                frames,
                ignore_index=True,
            )

            result["time"] = pd.to_datetime(result["time"])

            return result

        except requests.RequestException as e:

            wait = 10 * attempt

            print(f"Erreur réseau/API : {e}")

            if attempt < MAX_RETRIES:
                print(
                    f"Nouvelle tentative dans {wait}s..."
                )
                time.sleep(wait)

        except Exception as e:

            print(f"Erreur inattendue : {e}")

            if attempt < MAX_RETRIES:
                time.sleep(10 * attempt)

    print("Échec après plusieurs tentatives.")

    return None


# ============================================================
# TRANSFORMATION HORAIRE -> QUOTIDIEN
# ============================================================

def build_daily_features(hourly_df):
    """
    Transforme les données horaires en une ligne par jour.

    On ne construit PAS encore Tmax(t+1).
    Ici on crée simplement les features journalières.
    """

    df = hourly_df.copy()

    df["date"] = df["time"].dt.date

    # --------------------------------------------------------
    # Direction du vent
    # --------------------------------------------------------
    #
    # Une direction est cyclique :
    #
    #   359° ≈ 1°
    #
    # On ne doit donc surtout pas calculer naïvement
    # la moyenne des degrés.
    #
    # On transforme la direction en deux composantes :
    #
    # sin(theta), cos(theta)
    #

    direction_rad = (
        df["wind_direction_10m"]
        * math.pi
        / 180
    )

    df["wind_dir_sin"] = direction_rad.apply(math.sin)
    df["wind_dir_cos"] = direction_rad.apply(math.cos)

    # --------------------------------------------------------
    # Agrégation journalière
    # --------------------------------------------------------

    daily = (
        df.groupby(
            ["date", "lat", "lon"],
            as_index=False,
        )
        .agg(
            # ------------------------
            # Température
            # ------------------------

            temperature_max=(
                "temperature_2m",
                "max"
            ),

            temperature_min=(
                "temperature_2m",
                "min"
            ),

            temperature_mean=(
                "temperature_2m",
                "mean"
            ),

            # ------------------------
            # Humidité
            # ------------------------

            relative_humidity_mean=(
                "relative_humidity_2m",
                "mean"
            ),

            dew_point_mean=(
                "dew_point_2m",
                "mean"
            ),

            # ------------------------
            # Pression
            # ------------------------

            pressure_msl_mean=(
                "pressure_msl",
                "mean"
            ),

            # ------------------------
            # Nuages
            # ------------------------

            cloud_cover_mean=(
                "cloud_cover",
                "mean"
            ),

            # ------------------------
            # Rayonnement
            # ------------------------

            shortwave_radiation_hourly_sum=(
                "shortwave_radiation",
                "sum"
            ),

            # ------------------------
            # Vent
            # ------------------------

            wind_speed_mean=(
                "wind_speed_10m",
                "mean"
            ),

            wind_speed_max=(
                "wind_speed_10m",
                "max"
            ),

            wind_dir_sin_mean=(
                "wind_dir_sin",
                "mean"
            ),

            wind_dir_cos_mean=(
                "wind_dir_cos",
                "mean"
            ),

            # ------------------------
            # Pluie
            # ------------------------

            precipitation_sum=(
                "precipitation",
                "sum"
            ),

            # ------------------------
            # Sol
            # ------------------------

            soil_moisture_mean=(
                "soil_moisture_0_to_7cm",
                "mean"
            ),
        )
    )

    # --------------------------------------------------------
    # Conversion du rayonnement
    # --------------------------------------------------------
    #
    # Open-Meteo donne shortwave_radiation en W/m²,
    # moyenne de l'heure précédente.
    #
    # 1 W × 1 heure = 3600 J
    #
    # donc :
    #
    # W/m² * 3600 / 1e6 = MJ/m²
    #

    daily["shortwave_radiation_sum_MJ_m2"] = (
        daily["shortwave_radiation_hourly_sum"]
        * 3600
        / 1_000_000
    )

    daily = daily.drop(
        columns=["shortwave_radiation_hourly_sum"]
    )

    return daily


# ============================================================
# PETITS CHECKS
# ============================================================

def check_data(df, name):
    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    print()
    print("Shape:")
    print(df.shape)

    print()
    print("Premières lignes:")
    print(df.head())

    print()
    print("Dernières lignes:")
    print(df.tail())

    print()
    print("Valeurs manquantes:")
    print(df.isna().sum())

    print()
    print("Types:")
    print(df.dtypes)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Période demandée :")
    print(f"{START_DATE} -> {END_DATE}")
    print()

    print(
        f"Point demandé pour Nantes : "
        f"{NANTES[0]}, {NANTES[1]}"
    )

    # --------------------------------------------------------
    # 1. FETCH
    # --------------------------------------------------------

    hourly_df = fetch_batch([NANTES])

    if hourly_df is None:
        raise RuntimeError(
            "Impossible de récupérer les données Open-Meteo."
        )

    # --------------------------------------------------------
    # 2. Vérification des données horaires
    # --------------------------------------------------------

    check_data(
        hourly_df,
        "DATASET HORAIRE BRUT"
    )

    # --------------------------------------------------------
    # 3. Affichage de la cellule ERA5 réellement utilisée
    # --------------------------------------------------------

    grid_lat = hourly_df["lat"].iloc[0]
    grid_lon = hourly_df["lon"].iloc[0]

    print()
    print("Cellule ERA5 réellement utilisée :")
    print(f"latitude  = {grid_lat}")
    print(f"longitude = {grid_lon}")

    # --------------------------------------------------------
    # 4. Sauvegarde RAW
    # --------------------------------------------------------

    hourly_df.to_csv(
        DATA_DIR / "nantes_weather_hourly.csv",
        index=False,
    )

    print()
    print(
        "Données horaires sauvegardées dans "
        "nantes_weather_hourly.csv"
    )

    # --------------------------------------------------------
    # 5. Agrégation quotidienne
    # --------------------------------------------------------

    daily_df = build_daily_features(hourly_df)

    check_data(
        daily_df,
        "DATASET QUOTIDIEN"
    )

    # --------------------------------------------------------
    # 6. Sauvegarde DAILY
    # --------------------------------------------------------

    daily_df.to_csv(
        DATA_DIR / "nantes_weather_daily.csv",
        index=False,
    )

    print()
    print(
        "Données journalières sauvegardées dans "
        "nantes_weather_daily.csv"
    )