"""Évaluation commune : fenêtres temporelles, MAE des pics et traces reproductibles."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from research_features import ROOT, protocol


def period_masks(frame, start, end, validation_days=90):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    val_start = start - pd.Timedelta(days=validation_days)
    dates = frame["target_date"]
    gap = pd.Timedelta(days=1)
    masks = {
        "fit": dates < val_start - gap,
        "validation": (dates >= val_start) & (dates < start - gap),
        "development": dates < start - gap,
        "evaluation": (dates >= start) & (dates < end),
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Une fenêtre temporelle est vide.")
    assert dates[masks["fit"]].max() < frame.loc[masks["validation"], "date"].min()
    assert dates[masks["development"]].max() < frame.loc[masks["evaluation"], "date"].min()
    return masks


def training_weights(frame, mode, threshold):
    if mode == "uniform":
        return np.ones(len(frame))
    if mode == "changes":
        # Seulement les labels d'entraînement, jamais ceux de validation ou du test.
        change = (frame["tmax_next_day"] - frame["temperature_max"]).abs()
        return 1 + 2 * change.ge(threshold).to_numpy(dtype=float)
    raise ValueError(f"Pondération inconnue : {mode}")


def fit_predict(frame, columns, masks, variant, config):
    forbidden = [c for c in columns if c.startswith("target_") or c == "tmax_next_day"]
    if forbidden:
        raise ValueError(f"Cibles présentes dans les entrées : {forbidden}")
    target_mode = variant.get("target", "direct")
    algorithm = variant.get("algorithm", "xgboost")
    weight_mode = variant.get("weighting", "uniform")
    y = frame["tmax_next_day"].to_numpy(dtype=float)
    persistence = frame["temperature_max"].to_numpy(dtype=float)
    training_y = y - persistence if target_mode == "delta" else y.copy()
    if target_mode not in ["direct", "delta", "hourly24"]:
        raise ValueError(target_mode)
    X = frame[columns].to_numpy(dtype=float)
    fit, validation, development, evaluation = [masks[k].to_numpy() for k in ["fit", "validation", "development", "evaluation"]]
    weights_fit = training_weights(frame.loc[fit], weight_mode, config["extreme_change_threshold_c"])
    weights_development = training_weights(frame.loc[development], weight_mode, config["extreme_change_threshold_c"])

    if algorithm == "ridge":
        if target_mode == "hourly24":
            training_y = frame[[f"target_hour_{h:02d}" for h in range(24)]].to_numpy()
        estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        estimator.fit(X[development], training_y[development])
        prediction = estimator.predict(X[evaluation])
        if target_mode == "hourly24":
            prediction = prediction.max(axis=1)
        elif target_mode == "delta":
            prediction = prediction + persistence[evaluation]
        return prediction, {"algorithm": "ridge", "alpha": 10.0, "trees": None}

    version = tuple(int(part) for part in xgb.__version__.split(".")[:2])

    def fit_booster(labels):
        options = dict(config["xgboost"])
        fit_options = dict(eval_set=[(X[validation], labels[validation])], verbose=False)
        stopping = dict(eval_metric="mae", early_stopping_rounds=config["early_stopping_rounds"])
        if version >= (1, 6):
            options.update(stopping)
        else:
            fit_options.update(stopping)
        selection = xgb.XGBRegressor(**options)
        selection.fit(X[fit], labels[fit], sample_weight=weights_fit, **fit_options)
        count = int(selection.best_iteration + 1)
        final_options = dict(config["xgboost"], n_estimators=count)
        estimator = xgb.XGBRegressor(**final_options)
        estimator.fit(X[development], labels[development], sample_weight=weights_development)
        return estimator.predict(X[evaluation]), count

    if target_mode == "hourly24":
        predictions, trees = [], []
        for hour in range(24):
            prediction, count = fit_booster(frame[f"target_hour_{hour:02d}"].to_numpy(dtype=float))
            predictions.append(prediction)
            trees.append(count)
        prediction = np.column_stack(predictions).max(axis=1)
    else:
        prediction, trees = fit_booster(training_y)
        if target_mode == "delta":
            prediction = prediction + persistence[evaluation]
    return prediction, {"algorithm": "xgboost", "target": target_mode, "weighting": weight_mode, "trees": trees}


def error_metrics(frame, prediction, warm_threshold, change_threshold=4):
    truth = frame["tmax_next_day"].to_numpy(dtype=float)
    persistence = frame["temperature_max"].to_numpy(dtype=float)
    prediction = np.asarray(prediction)
    change = truth - persistence
    predicted_change = prediction - persistence
    error = np.abs(truth - prediction)
    large = np.abs(change) >= change_threshold
    cold = change <= -change_threshold
    warm = change >= change_threshold
    high = truth >= warm_threshold

    def subset_mean(mask):
        return float(error[mask].mean()) if mask.any() else np.nan

    return {
        "n": len(truth), "MAE_C": mean_absolute_error(truth, prediction),
        "RMSE_C": np.sqrt(mean_squared_error(truth, prediction)),
        "R2": r2_score(truth, prediction), "bias_C": float((prediction - truth).mean()),
        "n_large_changes": int(large.sum()), "MAE_large_changes_C": subset_mean(large),
        "n_cooling": int(cold.sum()), "MAE_cooling_C": subset_mean(cold),
        "n_warming": int(warm.sum()), "MAE_warming_C": subset_mean(warm),
        "large_change_direction_accuracy": float((np.sign(predicted_change[large]) == np.sign(change[large])).mean()) if large.any() else np.nan,
        "warm_threshold_C": warm_threshold,
        "n_warm_days": int(high.sum()), "MAE_warm_days_C": subset_mean(high),
        "predicted_change_std_C": float(predicted_change.std()),
        "observed_change_std_C": float(change.std()),
    }


def lag_diagnostic(frame, prediction, truth_lookup):
    """Alignement descriptif : vérité à date cible + décalage, jamais une correction."""
    candidates = {}
    for lag in [-1, 0, 1]:
        candidates[lag] = (frame["target_date"] + pd.Timedelta(days=lag)).map(truth_lookup).to_numpy()
    common = np.logical_and.reduce([np.isfinite(values) for values in candidates.values()])
    return {f"MAE_truth_date_plus_{lag}_C": mean_absolute_error(values[common], np.asarray(prediction)[common]) for lag, values in candidates.items()}


def block_gain_interval(persistence_error, model_error, repetitions=1000, block=7):
    """Intervalle descriptif du gain moyen par bootstrap circulaire de blocs de 7 jours."""
    difference = np.asarray(persistence_error) - np.asarray(model_error)
    rng = np.random.default_rng(42)
    starts = rng.integers(0, len(difference), size=(repetitions, int(np.ceil(len(difference) / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % len(difference)).reshape(repetitions, -1)[:, :len(difference)]
    return np.quantile(difference[indices].mean(axis=1), [0.025, 0.975])


def evaluate_suite(frame, variants, experiment_name, data_path):
    config = protocol()
    output = ROOT / "experiments/results" / experiment_name
    output.mkdir(parents=True, exist_ok=True)
    truth_lookup = pd.concat([
        frame[["date", "temperature_max"]].rename(columns={"date": "day", "temperature_max": "value"}),
        frame[["target_date", "tmax_next_day"]].rename(columns={"target_date": "day", "tmax_next_day": "value"}),
    ]).drop_duplicates("day").set_index("day")["value"]
    scores, predictions, lag_rows, model_records = [], [], [], []
    selected = None
    for window in config["evaluation_windows"]:
        masks = period_masks(frame, window["start"], window["end"], config["validation_days"])
        heldout = frame.loc[masks["evaluation"]]
        warm_threshold = float(frame.loc[masks["development"], "tmax_next_day"].quantile(config["warm_quantile_train_only"]))
        print(f"{experiment_name} — {window['name']} : {len(heldout)} jours", flush=True)
        fold_predictions = {"persistence": heldout["temperature_max"].to_numpy()}
        for name, variant in variants.items():
            prediction, metadata = fit_predict(frame, variant["columns"], masks, variant, config)
            fold_predictions[name] = prediction
            model_records.append({"window": window["name"], "variant": name, "features": len(variant["columns"]), **metadata})
        for name, prediction in fold_predictions.items():
            metrics = error_metrics(heldout, prediction, warm_threshold, config["extreme_change_threshold_c"])
            scores.append({"window": window["name"], "variant": name, **metrics})
            lag_rows.append({"window": window["name"], "variant": name, **lag_diagnostic(heldout, prediction, truth_lookup)})
            for date, truth, today, value in zip(heldout["target_date"], heldout["tmax_next_day"], heldout["temperature_max"], prediction):
                predictions.append({"window": window["name"], "variant": name, "target_date": date.isoformat(), "truth": float(truth), "today": float(today), "prediction": float(value)})
        # Le choix est figé à la fin des trois fenêtres 2025, avant les scores 2026.
        if window["name"] == config["evaluation_windows"][-2]["name"]:
            table = pd.DataFrame(scores)
            cv = table.loc[table["variant"].ne("persistence")]
            weighted = cv.assign(total_error=cv["MAE_C"] * cv["n"]).groupby("variant").agg(total_error=("total_error", "sum"), n=("n", "sum"))
            selected = (weighted["total_error"] / weighted["n"]).idxmin()
            print(f"Choix figé sur les fenêtres 2025 : {selected}", flush=True)
    scores = pd.DataFrame(scores)
    persistence_mae = scores.loc[scores["variant"].eq("persistence")].set_index("window")["MAE_C"]
    scores["gain_vs_persistence_C"] = scores["window"].map(persistence_mae) - scores["MAE_C"]
    predictions = pd.DataFrame(predictions)
    test_name = config["evaluation_windows"][-1]["name"]
    test_predictions = predictions.loc[predictions["window"].eq(test_name)]
    intervals = []
    for name, group in test_predictions.groupby("variant"):
        truth = group["truth"].to_numpy()
        baseline_error = np.abs(truth - group["today"].to_numpy())
        model_error = np.abs(truth - group["prediction"].to_numpy())
        low, high = block_gain_interval(baseline_error, model_error)
        intervals.append({"variant": name, "gain_low_C": low, "gain_high_C": high})
    provenance = {
        "experiment": experiment_name, "selected_on_cv": selected, "protocol": config,
        "data_file": str(Path(data_path).relative_to(ROOT)),
        "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "code_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__), ROOT / "scripts/research_features.py"]},
        "feature_sets": variants, "model_fits": model_records,
        "warning": "2026 was previously inspected; results are exploratory, not a prospective independent test.",
    }
    for name, table in [("scores", scores), ("predictions", predictions), ("lag_diagnostics", pd.DataFrame(lag_rows)), ("test_gain_intervals", pd.DataFrame(intervals))]:
        # to_json normalise NaN en null, conserve les résultats sans fichier CSV ignoré.
        (output / f"{name}.json").write_text(table.to_json(orient="records", indent=2, force_ascii=False) + "\n")
    (output / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n")
    return scores, predictions, pd.DataFrame(lag_rows), pd.DataFrame(intervals), selected, output


def summarize_cv(scores):
    cv = scores.loc[scores["window"].str.startswith("cv_")].copy()
    rows = []
    for name, group in cv.groupby("variant", sort=False):
        n_large = group["n_large_changes"].sum()
        rows.append({
            "variant": name,
            "MAE_C": np.average(group["MAE_C"], weights=group["n"]),
            "MAE_large_changes_C": np.average(group["MAE_large_changes_C"], weights=group["n_large_changes"]) if n_large else np.nan,
            "n": group["n"].sum(), "n_large_changes": n_large,
            "gain_vs_persistence_C": np.average(group["gain_vs_persistence_C"], weights=group["n"]),
        })
    return pd.DataFrame(rows).set_index("variant")
