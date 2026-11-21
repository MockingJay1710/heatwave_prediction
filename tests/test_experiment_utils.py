"""Contrôles des frontières temporelles et des dates calendaires."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from experiment_utils import (
    add_calendar_features, build_supervised, chronological_masks, prepare_daily_data,
)


def weather(dates):
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "lat": 47.25, "lon": -1.5,
        "temperature_max": np.arange(len(dates), dtype=float) + 10,
        "pressure_msl_mean": 1010.0, "relative_humidity_mean": 75.0,
    })


class CalendarPreparationTests(unittest.TestCase):
    def test_missing_day_is_not_tomorrow_or_yesterday(self):
        raw = weather(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
        paired = build_supervised(raw)
        self.assertEqual(paired["date"].dt.day.tolist(), [1, 4])
        self.assertEqual(paired["tmax_next_day"].tolist(), [11.0, 13.0])
        self.assertTrue(build_supervised(raw, include_history=True).empty)

    def test_history_uses_past_calendar_days_and_keeps_celsius(self):
        paired = build_supervised(weather(pd.date_range("2024-01-01", periods=6)), True)
        self.assertEqual(paired["date"].dt.day.tolist(), [3, 4, 5])
        self.assertEqual(paired["tmax_lag2"].tolist(), [10.0, 11.0, 12.0])
        self.assertEqual(paired["temperature_max"].tolist(), [12.0, 13.0, 14.0])
        self.assertEqual(paired["tmax_next_day"].tolist(), [13.0, 14.0, 15.0])

    def test_future_values_do_not_change_past_predictors(self):
        raw = weather(pd.date_range("2024-01-01", periods=8))
        changed = raw.copy()
        changed.loc[changed["date"] >= "2024-01-06", "temperature_max"] = 999.0
        before, after = build_supervised(raw, True), build_supervised(changed, True)
        columns = [c for c in before if c != "tmax_next_day"]
        past = before["date"] < "2024-01-06"
        pd.testing.assert_frame_equal(before.loc[past, columns], after.loc[past, columns])

    def test_cutoff_excludes_training_label_on_test_boundary(self):
        frame = build_supervised(weather(pd.date_range("2024-01-01", periods=8)))
        train, test = chronological_masks(frame, "2024-01-05")
        self.assertLess(frame.loc[train, "target_date"].max(), frame.loc[test, "date"].min())
        self.assertFalse(train.loc[frame["date"].eq("2024-01-04")].any())

    def test_calendar_uses_target_month_at_year_boundary(self):
        frame = build_supervised(weather(pd.date_range("2024-12-30", periods=4)))
        result = add_calendar_features(frame)
        january = result.loc[result["target_date"].eq("2025-01-01")].iloc[0]
        self.assertAlmostEqual(january["month_sin"], 0)
        self.assertAlmostEqual(january["month_cos"], 1)
        self.assertEqual(january["season_winter"], 1)
        self.assertTrue(result.filter(like="season_").sum(axis=1).eq(1).all())

    def test_rejects_duplicates_and_multiple_locations(self):
        raw = weather(pd.date_range("2024-01-01", periods=3))
        with self.assertRaisesRegex(ValueError, "Doublons"):
            prepare_daily_data(pd.concat([raw, raw.iloc[:1]]))
        raw.loc[1, "lat"] = 48.0
        with self.assertRaisesRegex(ValueError, "une seule cellule"):
            prepare_daily_data(raw)


if __name__ == "__main__":
    unittest.main()
