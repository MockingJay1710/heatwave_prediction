# Heatwave Lab

Le notebook historique `experiments/heatwave_linear_regression.ipynb` est conservé.
Les deux nouvelles expériences sont autonomes : leur code est écrit directement
dans les cellules, sans module d'expérimentation à suivre ailleurs.

| Notebook | Comparaison |
|---|---|
| [1 — Jour de l'année et saison](experiments/heatwave_xgboost_calendar.ipynb) | XGBoost météo seule contre météo + calendrier |
| [2 — Quatre voisins](experiments/heatwave_xgboost_neighbors.ipynb) | Modèle temporel contre le même modèle avec le voisinage |

Le deuxième notebook ajoute quatre points au nord, au sud, à l'est et à l'ouest,
à environ 75–85 km. Il utilise leurs températures, pressions et vents, les écarts
de température avec Nantes, et deux gradients de pression en hPa/km.
Les composantes du vent u et v sont calculées heure par heure puis moyennées par jour.

Les deux expériences ont la même cible : Tmax du lendemain. Toutes les entrées
viennent du jour courant ou de son calendrier connu à l'avance. Les journées
sont en UTC ; la prévision suppose les observations du jour terminées.
La séparation est chronologique : les premiers 80 % des jours pour apprendre,
les derniers 20 % pour évaluer. Les paramètres sont identiques et fixés.
La persistance (Tmax demain = Tmax aujourd'hui) sert de référence.
Les dates du test ayant déjà été examinées, les résultats restent exploratoires.
ERA5 est une réanalyse disponible avec retard, pas une prévision en temps réel.

## Utilisation

Depuis la racine du projet :

```bash
nix develop
python scripts/download_four_neighbors.py
jupyter lab experiments
```

Choisir le noyau **Python (STASC Nix)** et exécuter les cellules dans l'ordre.
La collecte crée `data/weather_five_points_daily.csv`, commun aux deux notebooks,
sans modifier les anciens CSV. Les données et les caches restent ignorés par Git.

Pour exécuter les deux notebooks sans ouvrir Jupyter :

```bash
nix develop --command python scripts/execute_notebooks.py \
  experiments/heatwave_xgboost_calendar.ipynb \
  experiments/heatwave_xgboost_neighbors.ipynb
```

Les résultats et graphiques sont enregistrés dans les notebooks.
