import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fetch_spatial_data import study_points
from research_features import (
    calendar_columns, fit_gradient, local_tables, protocol,
    spatial_columns, validate_hourly, wind_components,
)
from research_evaluation import period_masks, training_weights


def sample_hourly(days=10):
    rows = []
    config = protocol()
    center_lat = config["center"]["latitude"]
    center_lon = config["center"]["longitude"]
    for point in study_points(config):
        x = 6371008.8 * np.cos(np.deg2rad(center_lat)) * np.deg2rad(point["longitude"] - center_lon)
        y = 6371008.8 * np.deg2rad(point["latitude"] - center_lat)
        for timestamp in pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC"):
            rows.append({
                "point": point["point"], "time": timestamp,
                "lat": point["latitude"], "lon": point["longitude"],
                "temperature_2m": 20 + x * 1e-5 + y * 2e-5 + timestamp.hour * 0.1,
                "pressure_msl": 1010 + x * 2e-5 - y * 1e-5,
                "wind_speed_10m": 5.0, "wind_direction_10m": 270.0,
                "relative_humidity_2m": 70.0, "dew_point_2m": 10.0,
                "cloud_cover": 40.0, "shortwave_radiation": 100.0,
                "precipitation": 0.0, "soil_moisture_0_to_7cm": 0.3,
            })
    return pd.DataFrame(rows)


class ResearchProtocolTests(unittest.TestCase):
    def test_meteorological_wind_direction(self):
        u, v = wind_components(np.array([5, 5, 5, 5]), np.array([0, 90, 180, 270]))
        np.testing.assert_allclose(u, [0, -5, 0, 5], atol=1e-12)
        np.testing.assert_allclose(v, [-5, 0, 5, 0], atol=1e-12)

    def test_gradient_reconstructs_linear_field(self):
        coords = np.array([[0, 100], [100, 0], [0, -100], [-100, 0]])
        values = np.array([[2, 1, -2, -1], [4, 2, -4, -2]])
        np.testing.assert_allclose(fit_gradient(values, coords), [[0.01, 0.02], [0.02, 0.04]])

    def test_gradient_advection_units_and_upwind_choice(self):
        hourly = validate_hourly(sample_hourly(2))
        features, _, _ = spatial_columns(hourly)
        for scale in ["local", "regional"]:
            np.testing.assert_allclose(features[f"physics_{scale}_gradient_temperature_x"], 1e-5)
            np.testing.assert_allclose(features[f"physics_{scale}_gradient_temperature_y"], 2e-5)
            np.testing.assert_allclose(features[f"physics_{scale}_thermal_advection_K_per_hour"], -0.18)
            self.assertTrue(features[f"upwind_{scale}_temperature_contrast"].lt(0).all())

    def test_rejects_incomplete_utc_day(self):
        with self.assertRaises(ValueError):
            validate_hourly(sample_hourly(2).iloc[1:])

    def test_all_features_precede_issue_and_targets_match(self):
        frame, _, _ = local_tables(validate_hourly(sample_hourly(5)))
        self.assertTrue(frame["last_observation_time"].lt(frame["issue_time"]).all())
        np.testing.assert_allclose(frame[[f"target_hour_{h:02d}" for h in range(24)]].max(axis=1), frame["tmax_next_day"])

    def test_future_observations_do_not_change_past_features(self):
        raw = sample_hourly(8)
        before, _, _ = local_tables(validate_hourly(raw))
        raw.loc[raw["time"] >= pd.Timestamp("2024-01-06", tz="UTC"), "temperature_2m"] = 1000
        after, _, _ = local_tables(validate_hourly(raw))
        columns = [c for c in before if not c.startswith("target_hour_") and c != "tmax_next_day"]
        mask = before["date"] < "2024-01-06"
        pd.testing.assert_frame_equal(before.loc[mask, columns], after.loc[mask, columns])

    def test_leap_year_annual_phase_wraps(self):
        frame = pd.DataFrame({"target_date": pd.to_datetime(["2024-01-01", "2024-12-31", "2025-01-01"])})
        calendar = calendar_columns(frame)
        self.assertEqual(calendar.loc[0, "calendar_day_sin"], 0)
        self.assertEqual(calendar.loc[2, "calendar_day_sin"], 0)
        self.assertGreater(calendar.loc[1, "calendar_day_cos"], 0.99)

    def test_fold_masks_purge_the_boundaries(self):
        targets = pd.date_range("2024-01-01", "2025-08-01")
        frame = pd.DataFrame({"target_date": targets, "date": targets - pd.Timedelta(days=1)})
        masks = period_masks(frame, "2025-01-01", "2025-05-01")
        self.assertLess(frame.loc[masks["fit"], "target_date"].max(), frame.loc[masks["validation"], "date"].min())
        self.assertLess(frame.loc[masks["development"], "target_date"].max(), frame.loc[masks["evaluation"], "date"].min())
        self.assertFalse((masks["development"] & masks["evaluation"]).any())

    def test_weights_use_changes_not_absolute_temperature(self):
        frame = pd.DataFrame({"temperature_max": [10, 35, 20], "tmax_next_day": [15, 35, 14]})
        np.testing.assert_equal(training_weights(frame, "changes", 4), [3, 1, 3])


if __name__ == "__main__":
    unittest.main()
