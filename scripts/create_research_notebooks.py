"""Créer les notebooks du protocole ; l'exécution et les résultats restent séparés."""

import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": (textwrap.dedent(text).strip() + "\n").splitlines(keepends=True)}


DESIGNS = {
    "01_calendar_diagnostics": {
        "title": "1 — Calendrier, pics et décalage des prédictions",
        "question": "Le mois ou la position précise dans l'année apporte-t-il de l'information au-delà de la météo quotidienne locale ? Les prévisions ressemblent-elles davantage à la veille qu'au vrai lendemain ?",
        "spatial": False,
        "variants": '''
            daily = groups["daily"]
            variants = {
                "xgb_daily": {"columns": daily},
                "xgb_month_raw": {"columns": daily + ["calendar_month"]},
                "xgb_day_raw": {"columns": daily + ["calendar_day_of_year"]},
                "xgb_month_cycle": {"columns": daily + ["calendar_month_sin", "calendar_month_cos"]},
                "xgb_day_cycle": {"columns": daily + ["calendar_day_sin", "calendar_day_cos"]},
                "xgb_season": {"columns": daily + ["calendar_winter", "calendar_spring", "calendar_summer", "calendar_autumn"]},
                "xgb_year": {"columns": daily + ["calendar_year"]},
                "ridge_daily": {"columns": daily, "algorithm": "ridge"},
            }
        ''',
        "method": '''
            Chaque variable calendaire est testée **séparément**, à partir de la date cible connue à l'avance.
            Le jour de l'année cyclique utilise la fraction de l'année civile : 365 jours ou 366 selon
            l'année. Le 1er janvier correspond à la même phase chaque année. Le mois cyclique utilise
            sinus/cosinus ; la saison utilise quatre indicatrices. L'année est un contrôle exploratoire,
            fragile avec ce petit nombre d'années ; aucun jour du mois n'est ajouté.

            Le diagnostic de décalage compare chaque prévision datée D à la vérité aux dates D−1, D et D+1,
            sur le même sous-ensemble de dates. Une persistance reproduit exactement D−1 : un décalage apparent
            ne prouve donc pas un mauvais alignement. On ne décale jamais les prévisions pour améliorer les scores.
            Les coefficients et la standardisation ne concernent ici que le témoin Ridge : pipeline ajusté
            sur l'entraînement, avec alpha fixé à 10.
        ''',
    },
    "02_hourly_information": {
        "title": "2 — Les moyennes journalières masquent-elles la dynamique ?",
        "question": "Les valeurs horaires et les changements en fin de journée améliorent-ils la prévision, à algorithme et dates inchangés ?",
        "spatial": False,
        "variants": '''
            daily = groups["daily"]
            with_hourly = daily + groups["hourly"]
            variants = {
                "xgb_daily": {"columns": daily},
                "xgb_daily_hourly": {"columns": with_hourly},
                "ridge_daily": {"columns": daily, "algorithm": "ridge"},
                "ridge_daily_hourly": {"columns": with_hourly, "algorithm": "ridge"},
            }
        ''',
        "method": '''
            On conserve les agrégats quotidiens et les historiques J−1/J−2 de Nantes. On ajoute
            température, pression, humidité et composantes u/v du vent à 06 h, 12 h, 18 h et 23 h UTC de J.
            On mesure aussi leurs changements de fin de journée sur 6 et 12 h (23−17 h et 23−11 h,
            pour température, pression et u/v). Ces 28 nouvelles colonnes ne requièrent aucune donnée de J+1.

            Le calendrier est absent de cette ablation pour isoler l'apport des observations horaires.
            La comparaison XGBoost/Ridge sert à distinguer un effet d'information d'un effet du modèle.
            Le calendrier et l'heure sont en UTC ; ne pas comparer directement ces Tmax aux anciens jours
            en heure locale. Les nouvelles variantes partagent toutes la même définition de la cible.
        ''',
    },
    "03_spatial_information": {
        "title": "3 — Voisins, gradients et information en amont du vent",
        "question": "Les cellules situées à environ 75 ou 200 km permettent-elles de mieux anticiper les variations que les seules observations locales, même horaires ?",
        "spatial": True,
        "variants": '''
            baseline = groups["daily"] + groups["hourly"]
            neighbors = groups["local_raw"] + groups["regional_raw"]
            variants = {
                "xgb_local_hourly": {"columns": baseline},
                "xgb_neighbors_75km": {"columns": baseline + groups["local_raw"]},
                "xgb_neighbors_200km": {"columns": baseline + groups["regional_raw"]},
                "xgb_neighbors_both": {"columns": baseline + neighbors},
                "xgb_gradients": {"columns": baseline + groups["physics"]},
                "xgb_gradients_upwind": {"columns": baseline + groups["physics"] + groups["upwind"]},
                "xgb_neighbors_physics": {"columns": baseline + neighbors + groups["physics"] + groups["upwind"]},
            }
        ''',
        "method": '''
            Le script distinct `scripts/fetch_spatial_data.py` collecte Nantes et quatre points cardinaux
            à chacune des deux distances indicatives, y compris à l'ouest en mer. On utilise `models=era5`,
            `cell_selection=nearest`, sans correction d'altitude (`elevation=nan`). Les coordonnées réellement
            renvoyées sont enregistrées ; les doublons de cellules et les journées incomplètes sont refusés.
            L'ancien dataset n'est pas modifié.

            Pour chaque voisin : température et pression moyennes de J, puis température, pression,
            humidité et vent u/v à 23 h. Les jointures portent sur date et cellule avant passage en colonnes.
            Les gradients de température (K/m) et pression (hPa/m) sont estimés par un plan local sur les
            contrastes avec le centre, en tenant compte des positions réellement retournées (distances en mètres).

            Pour une direction météorologique d'origine θ : u = −V sin θ et v = −V cos θ.
            L'indicateur d'advection est −(u ∂T/∂x + v ∂T/∂y) × 3600, en K/h, calculé à 23 h.
            Il ne représente que le transport horizontal de surface, pas toute la physique de Tmax.
            Le point amont est le cardinal le plus opposé au vecteur vitesse. On ajoute ses contrastes de
            température et de pression, et une durée indicative distance/vitesse. Pour un vent < 0,5 m/s,
            ces indicateurs sont neutralisés et un drapeau de calme est fourni. Ce n'est pas une trajectoire.
        ''',
    },
    "04_changes_and_weighting": {
        "title": "4 — Prédire une variation et donner du poids aux changements rares",
        "question": "Le lissage des pics vient-il aussi de la cible et de la pondération, indépendamment de l'information disponible ?",
        "spatial": True,
        "variants": '''
            local = groups["daily"] + groups["hourly"]
            spatial = local + groups["local_raw"] + groups["regional_raw"] + groups["physics"] + groups["upwind"]
            variants = {
                "xgb_local_direct": {"columns": local},
                "xgb_local_delta": {"columns": local, "target": "delta"},
                "xgb_spatial_direct": {"columns": spatial},
                "xgb_spatial_delta": {"columns": spatial, "target": "delta"},
                "xgb_spatial_delta_weighted": {"columns": spatial, "target": "delta", "weighting": "changes"},
                "ridge_spatial_delta": {"columns": spatial, "target": "delta", "algorithm": "ridge"},
            }
        ''',
        "method": '''
            On compare Tmax(J+1) directement à ΔTmax = Tmax(J+1) − Tmax(J).
            Pour la seconde cible, la prévision finale est Tmax(J) + variation prédite.
            Ainsi le niveau connu peut être réutilisé sans obliger les arbres à apprendre toute l'échelle
            des températures absolues. La persistance correspond à une variation prédite nulle.

            On croise les deux cibles avec deux jeux d'entrées fixés : local + horaire, puis le même jeu
            enrichi de tous les voisins et indicateurs physiques. Aucun jeu n'est choisi à partir du test.
            Une variante pondère par 3 les observations **d'entraînement** dont |ΔTmax| ≥ 4 °C ; toutes les
            autres gardent un poids de 1. Les scores et la validation restent non pondérés.
            Cette pondération peut aider les variations rares tout en dégradant la MAE globale : on montre les deux.
            Ridge avec alpha fixé complète le contrôle algorithmique, sans optimisation sur 2026.
        ''',
    },
    "05_hourly_targets": {
        "title": "5 — Prédire les 24 températures du lendemain puis leur maximum",
        "question": "Une décomposition de la cible en 24 heures aide-t-elle davantage qu'une prévision directe de Tmax ?",
        "spatial": True,
        "variants": '''
            features = groups["daily"] + groups["hourly"] + groups["physics"] + groups["upwind"]
            variants = {
                "xgb_direct_tmax": {"columns": features},
                "xgb_24h_then_max": {"columns": features, "target": "hourly24"},
                "ridge_direct_tmax": {"columns": features, "algorithm": "ridge"},
                "ridge_24h_then_max": {"columns": features, "algorithm": "ridge", "target": "hourly24"},
            }
        ''',
        "method": '''
            Les entrées sont fixées : agrégats locaux, dynamique horaire, gradients et amont du vent.
            On apprend soit une Tmax, soit les 24 températures horaires UTC de J+1 ; le maximum de
            ces 24 prévisions devient la Tmax prédite. Les cibles horaires sont construites par jointure
            sur la date cible et contrôlées contre la Tmax réelle ; elles ne figurent jamais dans les entrées.

            Pour XGBoost, on entraîne 24 régresseurs, chacun avec son propre arrêt anticipé sur la
            validation interne, puis son réentraînement sur la période de développement. Ridge gère
            les 24 sorties dans le même pipeline. Aucun modèle ne reçoit les heures déjà observées de J+1.
            Cette approche ne garantit pas une courbe physiquement cohérente et le maximum est non linéaire :
            l'expérience doit juger son effet, sans supposer une amélioration.
        ''',
    },
}


def build(name):
    design = DESIGNS[name]
    cells = [
        markdown(f"# {design['title']}\n\n**Question :** {design['question']}"),
        markdown('''
            ## Protocole commun

            Prévision émise à **00 h UTC du jour cible** : toutes les entrées proviennent de la veille
            ou de jours antérieurs, avec une dernière observation à 23 h UTC. La cible est le maximum
            des 24 températures horaires du jour cible UTC. ERA5 est une réanalyse disponible avec retard :
            cette étude rétrospective ne simule pas la disponibilité d'un service de prévision en direct.

            Le fichier `experiments/protocol.json` fixe les hypothèses et les paramètres. On évalue trois
            fenêtres successives couvrant 2025, avec entraînement croissant. Dans chaque fenêtre,
            les 90 derniers jours de développement servent à l'arrêt anticipé ; une journée est purgée
            aux frontières. On réentraîne ensuite sur le développement avec le nombre d'arbres retenu.
            Le choix entre variantes se fait sur la MAE pondérée par les effectifs des fenêtres 2025,
            **avant** l'évaluation sur janvier–septembre 2026. La persistance reste une référence séparée.

            L'année 2026 ayant déjà été consultée auparavant, ses résultats sont exploratoires.
            Les modifications du protocole doivent produire une nouvelle expérience et un nouveau commit.
            Aucun hyperparamètre n'est ajusté pour obtenir un meilleur score sur 2026.
        '''),
        code('''
            from pathlib import Path
            import sys
            import json
            import numpy as np
            import pandas as pd
            from IPython.display import display

            PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "flake.nix").exists())
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from research_features import load_research_data, protocol
            from research_evaluation import evaluate_suite, summarize_cv
            from research_reporting import report, plot_cells
            pd.set_option("display.max_columns", 20)
        '''),
        markdown("## Variables et hypothèse\n\n" + textwrap.dedent(design["method"]).strip()),
        code(f'''
            frame, groups, cells, data_path = load_research_data(spatial={design['spatial']})
            print(f"{{len(frame)}} paires : {{frame['target_date'].min().date()}} → {{frame['target_date'].max().date()}}")
            display(pd.DataFrame([{{"groupe": k, "nombre_variables": len(v)}} for k, v in groups.items()]))
            display(frame[["date", "target_date", "issue_time", "last_observation_time", "temperature_max", "tmax_next_day"]].head())
            assert (frame["last_observation_time"] < frame["issue_time"]).all()
        '''),
        code(design["variants"]),
        markdown('''
            ## Expérience et enregistrement

            Le code commun est dans `scripts/research_features.py` et `scripts/research_evaluation.py`.
            Chaque variante utilise les mêmes lignes ; les colonnes et les paramètres sont explicitement
            enregistrés avec les nombres d'arbres, les empreintes du dataset et du code.
            Les températures restent en °C. XGBoost ne reçoit pas de standardisation globale ; Ridge
            ajuste son standardiseur uniquement sur la période d'entraînement de chaque fenêtre.
        '''),
        code(f'''
            scores, predictions, lag_scores, intervals, selected, output = evaluate_suite(
                frame, variants, "{name}", data_path
            )
        '''),
        markdown('''
            ## Résultats : niveau, variations et pics

            La MAE globale est la métrique principale. Les fortes variations sont définies **avant** les résultats
            par |Tmax(J+1) − Tmax(J)| ≥ 4 °C, avec refroidissements et réchauffements séparés.
            Les journées chaudes utilisent le 90e percentile des cibles du développement de chaque fenêtre,
            jamais un seuil calculé sur le test. Il ne s'agit pas d'une définition de canicule.

            Le gain positif signifie une erreur inférieure à la persistance. Les intervalles descriptifs
            par bootstrap de blocs de sept jours tiennent partiellement compte de l'autocorrélation,
            sans garantir la généralisation à d'autres années ni corriger les comparaisons multiples.
        '''),
        code('report(scores, predictions, lag_scores, intervals, selected, output)'),
    ]
    if design["spatial"]:
        cells += [markdown("## Géométrie du voisinage\n\nCoordonnées des cellules réellement retournées, sans présumer une station de mesure."), code('display(cells)\nplot_cells(cells, output)')]
    cells += [
        markdown('''
            ## Interprétation et limites

            Un gain global sans gain sur les refroidissements ne valide pas l'hypothèse d'anticipation des fronts.
            À l'inverse, un gain sur quelques pics peut être instable : regarder les effectifs, les trois fenêtres
            2025 et les mois de 2026. Ne pas confondre mauvaise information, décalage d'alignement et faible
            extrapolation du modèle. Cette étude ne mesure pas directement les fronts ni les masses d'air.
            L'absence de gain avec ces variables ne prouve pas que toute information spatiale est inutile.

            Avec moins de trois années et environ mille journées, les réseaux neuronaux ne sont pas la prochaine
            étape automatique. Une confirmation sur des années supplémentaires et des prévisions archivées
            disponibles à l'heure d'émission sera nécessaire avant une conclusion opérationnelle.

            Sources : [Open-Meteo, réanalyse et paramètres](https://open-meteo.com/en/docs/historical-weather-api),
            [évaluation temporelle](https://scikit-learn.org/stable/modules/cross_validation.html),
            [complexité et régularisation XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html).
        '''),
        code('''
            cv = summarize_cv(scores)
            test = scores.loc[scores["window"].eq("test_2026")].set_index("variant")
            print(f"Variante choisie en 2025 : {selected}")
            print(f"MAE 2025 : {cv.loc[selected, 'MAE_C']:.3f} °C ; persistance : {cv.loc['persistence', 'MAE_C']:.3f} °C")
            print(f"MAE 2026 : {test.loc[selected, 'MAE_C']:.3f} °C ; persistance : {test.loc['persistence', 'MAE_C']:.3f} °C")
            print(f"MAE 2026 sur fortes variations : {test.loc[selected, 'MAE_large_changes_C']:.3f} °C ; n = {test.loc[selected, 'n_large_changes']}")
            print(f"Traces et figures enregistrées dans {output.relative_to(PROJECT_ROOT)}")
        '''),
    ]
    for i, cell in enumerate(cells):
        cell["id"] = f"{name[:12]}-{i:02d}"
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python (STASC Nix)", "language": "python", "name": "stasc"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path = ROOT / "experiments/research" / f"{name}.ipynb"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(path.relative_to(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", choices=list(DESIGNS))
    args = parser.parse_args()
    for name in args.names or DESIGNS:
        build(name)


if __name__ == "__main__":
    main()
