"""Tableaux et figures communs aux notebooks d'expérimentation."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from research_evaluation import summarize_cv


def report(scores, predictions, lag_scores, intervals, selected, output):
    cv = summarize_cv(scores)
    test = scores.loc[scores["window"].eq("test_2026")].set_index("variant")
    print("Moyenne pondérée des trois fenêtres 2025 (choix du modèle) :")
    display(cv.round(3))
    print(f"Choix sur 2025, maintenu pour 2026 : {selected}")
    print("Test descriptif 2026 — MAE en °C et effectifs :")
    display(test[["MAE_C", "RMSE_C", "bias_C", "n", "n_large_changes", "MAE_large_changes_C", "MAE_cooling_C", "MAE_warming_C", "MAE_warm_days_C", "gain_vs_persistence_C"]].round(3))
    print("Gain vs persistance : intervalle descriptif par blocs de 7 jours (sans correction pour comparaisons multiples) :")
    display(intervals.set_index("variant").round(3))
    names = list(cv.index)
    labels = [name.replace("_", " ") for name in names]
    positions = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(17, max(5, len(names) * 0.5)))
    for ax, table, column, title in [
        (axes[0], cv, "MAE_C", "Validation glissante — année 2025"),
        (axes[1], test, "MAE_C", "Évaluation 2026 — tous les jours"),
        (axes[2], test, "MAE_large_changes_C", "Évaluation 2026 — |ΔTmax| ≥ 4 °C"),
    ]:
        ax.barh(positions, table.loc[names, column], color=["#777777" if n == "persistence" else "#20887a" if n == selected else "#568fc4" for n in names])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set(xlabel="MAE (°C), plus faible = mieux", title=title)
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "scores.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    test_predictions = predictions.loc[predictions["window"].eq("test_2026")]
    chosen = test_predictions.loc[test_predictions["variant"].eq(selected)].copy()
    chosen["target_date"] = pd.to_datetime(chosen["target_date"])
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    axes[0].plot(chosen["target_date"], chosen["truth"], color="black", label="Tmax réelle", linewidth=1.3)
    axes[0].plot(chosen["target_date"], chosen["today"], alpha=0.5, label="Persistance")
    axes[0].plot(chosen["target_date"], chosen["prediction"], alpha=0.8, label=selected)
    axes[0].set(ylabel="Tmax (°C)", title="Modèle retenu en 2025 — prévisions 2026")
    axes[0].legend()
    axes[1].scatter(chosen["truth"] - chosen["today"], chosen["prediction"] - chosen["today"], alpha=0.45)
    limits = [min((chosen["truth"] - chosen["today"]).min(), -6), max((chosen["truth"] - chosen["today"]).max(), 6)]
    axes[1].plot(limits, limits, linestyle="--", color="black", label="Changement parfaitement anticipé")
    axes[1].axhline(0, color="grey", alpha=0.4)
    axes[1].set(xlabel="Variation réelle Tmax(J+1) − Tmax(J), °C", ylabel="Variation prédite, °C", title="Anticipation des retournements")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "dynamics.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print("Diagnostic de décalage : vérité comparée à date_cible + décalage. -1 = la veille.")
    display(lag_scores.loc[lag_scores["window"].eq("test_2026")].set_index("variant").drop(columns="window").round(3))
    print("Un minimum à -1 signale une ressemblance avec la veille ; il ne prouve pas une erreur de dates.")
    monthly = test_predictions.copy()
    monthly["month"] = pd.to_datetime(monthly["target_date"]).dt.to_period("M").astype(str)
    monthly["absolute_error"] = (monthly["truth"] - monthly["prediction"]).abs()
    print("MAE par mois cible, mêmes jours pour tous les modèles :")
    display(monthly.pivot_table(index="month", columns="variant", values="absolute_error", aggfunc="mean").round(3))
    print("Les effectifs des fortes variations sont limités ; pas de conclusion causale sur les fronts météorologiques.")


def plot_cells(cells, output):
    cells = cells.reset_index() if "point" not in cells.columns else cells.copy()
    center = cells.loc[cells["point"].eq("center")].iloc[0]
    x = 6371.0088 * np.cos(np.deg2rad(center["lat"])) * np.deg2rad(cells["lon"] - center["lon"])
    y = 6371.0088 * np.deg2rad(cells["lat"] - center["lat"])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(x, y, s=60, color=["black" if p == "center" else "#20887a" if p.startswith("local") else "#568fc4" for p in cells["point"]])
    for px, py, label in zip(x, y, cells["point"]):
        ax.annotate(label, (px, py), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Distance vers l'est (km)", ylabel="Distance vers le nord (km)", title="Cellules ERA5 réellement utilisées — centre et deux échelles")
    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "cells.png", dpi=140, bbox_inches="tight")
    plt.show()
    plt.close(fig)
