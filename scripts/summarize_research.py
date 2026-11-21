"""Résumer les expériences exécutées sans réentraîner ni choisir sur le test."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_evaluation import block_gain_interval, summarize_cv

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
PAIRS = {
    "01_calendar_diagnostics": [("xgb_daily", "xgb_month_cycle"), ("xgb_daily", "xgb_day_cycle")],
    "02_hourly_information": [("xgb_daily", "xgb_daily_hourly"), ("ridge_daily", "ridge_daily_hourly")],
    "03_spatial_information": [("xgb_local_hourly", "xgb_neighbors_200km"), ("xgb_local_hourly", "xgb_gradients_upwind")],
    "04_changes_and_weighting": [
        ("xgb_local_direct", "xgb_local_delta"),
        ("xgb_spatial_direct", "xgb_spatial_delta"),
        ("xgb_local_delta", "xgb_spatial_delta"),
        ("xgb_spatial_delta", "xgb_spatial_delta_weighted"),
    ],
    "05_hourly_targets": [("xgb_direct_tmax", "xgb_24h_then_max"), ("ridge_direct_tmax", "ridge_24h_then_max")],
}


def main():
    rows, pairs = [], []
    for name, contrasts in PAIRS.items():
        directory = RESULTS / name
        scores = pd.read_json(directory / "scores.json")
        provenance = json.loads((directory / "provenance.json").read_text())
        selected = provenance["selected_on_cv"]
        cv = summarize_cv(scores)
        test = scores.loc[scores["window"].eq("test_2026")].set_index("variant")
        rows.append({
            "experiment": name, "selected_on_2025": selected,
            "cv_2025_MAE_C": cv.loc[selected, "MAE_C"],
            "test_2026_MAE_C": test.loc[selected, "MAE_C"],
            "test_persistence_MAE_C": test.loc["persistence", "MAE_C"],
            "test_large_change_MAE_C": test.loc[selected, "MAE_large_changes_C"],
            "test_large_changes_n": int(test.loc[selected, "n_large_changes"]),
            "test_n": int(test.loc[selected, "n"]),
        })
        predictions = pd.read_json(directory / "predictions.json")
        predictions = predictions.loc[predictions["window"].eq("test_2026")]
        for baseline, added in contrasts:
            left = predictions.loc[predictions["variant"].eq(baseline)].set_index("target_date").sort_index()
            right = predictions.loc[predictions["variant"].eq(added)].set_index("target_date").sort_index()
            if not left.index.equals(right.index) or not np.allclose(left["truth"], right["truth"]):
                raise ValueError("Les modèles comparés n'ont pas les mêmes jours et cibles.")
            baseline_error = (left["truth"] - left["prediction"]).abs().to_numpy()
            added_error = (right["truth"] - right["prediction"]).abs().to_numpy()
            low, high = block_gain_interval(baseline_error, added_error)
            pairs.append({
                "experiment": name, "baseline": baseline, "comparison": added,
                "test_MAE_gain_C": float((baseline_error - added_error).mean()),
                "gain_low_C": low, "gain_high_C": high,
                "cv_MAE_gain_C": float(cv.loc[baseline, "MAE_C"] - cv.loc[added, "MAE_C"]),
                "test_large_change_MAE_gain_C": float(test.loc[baseline, "MAE_large_changes_C"] - test.loc[added, "MAE_large_changes_C"]),
            })
    summary, contrasts = pd.DataFrame(rows), pd.DataFrame(pairs)
    (RESULTS / "summary.json").write_text(summary.to_json(orient="records", indent=2, force_ascii=False) + "\n")
    (RESULTS / "paired_comparisons.json").write_text(contrasts.to_json(orient="records", indent=2, force_ascii=False) + "\n")
    print(summary.to_string(index=False))
    print("\nComparaisons appariées — gain positif = amélioration :")
    print(contrasts.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
