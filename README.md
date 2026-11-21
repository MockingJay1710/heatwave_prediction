# Heatwave Lab

```text
heatwave-lab/
├── data/          # Hourly and daily weather CSVs
├── experiments/   # Your notebook and its saved plots
├── scripts/       # Your data collection script
└── notes/         # Your existing notes and PDFs
```

Your original Nix environment and Git setup stay at the project root.

## Continue your experiments

From `heatwave-lab/`:

```bash
nix develop
jupyter lab experiments
```

Open `heatwave_linear_regression.ipynb`. It reads from `../data/` and saves
plots alongside the notebook. Duplicate the notebook when you want to try a
separate experiment.

Your collection script is in `scripts/heatwave_data.py`; it now saves CSVs
in `data/`. Only file paths were adjusted; the existing model logic is unchanged.
CSV files and local caches are ignored by Git.

## Étude des informations disponibles

Les essais organisés sont dans `experiments/research/`. Les notebooks racine
conservent les explorations historiques et leurs anciennes coupures. Le notebook
`heatwave_linear_regression.ipynb` contient des essais interactifs, dont une
standardisation globale ; ses scores ne servent pas de référence à cette étude.
Les nouveaux témoins sont réentraînés avec un protocole commun et sans cette fuite.

| Notebook | Question étudiée |
|---|---|
| [01 — Calendrier](experiments/research/01_calendar_diagnostics.ipynb) | Mois, jour de l'année, encodages cycliques, saison et décalage apparent |
| [02 — Informations horaires](experiments/research/02_hourly_information.ipynb) | Apport des valeurs horaires et tendances des dernières heures |
| [03 — Voisinage](experiments/research/03_spatial_information.ipynb) | Voisins à 75/200 km, gradients, advection et conditions en amont |
| [04 — Variations](experiments/research/04_changes_and_weighting.ipynb) | Cible en variation et pondération des changements rares |
| [05 — Cibles horaires](experiments/research/05_hourly_targets.ipynb) | Prévoir les 24 températures horaires, puis leur maximum |

Le protocole est fixé dans `experiments/protocol.json` : émission à **00 h UTC**,
dernière observation la veille à 23 h, cible sur les 24 heures UTC du jour suivant.
Les agrégats quotidiens et les variables horaires sont reconstruits à partir des
mêmes observations. Les cibles J+1 et les historiques sont alignés par dates exactes.
La standardisation de Ridge est ajustée sur l'entraînement ; XGBoost utilise les
unités physiques. Les cibles et les métadonnées ne sont jamais des prédicteurs.

On compare trois fenêtres successives couvrant 2025, avec entraînement croissant.
Une validation interne de 90 jours sert à l'arrêt anticipé, avec une journée purgée
aux frontières. Le modèle est ensuite réentraîné sur le développement avec le
nombre d'arbres retenu. Le choix entre variantes se fait sur les scores 2025,
avant l'évaluation janvier–septembre 2026. La persistance est conservée séparément.
Les dates 2026 ayant déjà été examinées, les résultats restent exploratoires.

Les métriques comprennent MAE, RMSE, biais, erreurs sur |ΔTmax| ≥ 4 °C,
refroidissements, réchauffements et journées chaudes. Le seuil des journées chaudes
est le 90e percentile du développement de chaque fenêtre ; ce n'est pas une
définition de canicule. Les diagnostics de décalage ne modifient pas les prédictions.
Les intervalles de gain par blocs de sept jours sont descriptifs, sans correction
pour les comparaisons multiples.

Chaque expérience enregistre ses scores, prédictions, figures, variables,
paramètres, nombres d'arbres et empreintes SHA-256 dans
`experiments/results/<experience>/`. Le manifeste de collecte est conservé dans
`experiments/results/dataset_manifest.json`. Les observations brutes restent ignorées.

## Résultats de cette exécution

Neuf cellules distinctes et 215 352 observations horaires ont été collectées.
Les cinq expériences utilisent 994 paires de jours, dont 365 pour les trois
fenêtres d'évaluation 2025 et 266 pour l'évaluation 2026. Cette dernière comporte
41 variations de Tmax d'au moins 4 °C en valeur absolue.

Le tableau présente la variante choisie **sur 2025** dans chaque expérience,
sans la remplacer par celle qui obtient le meilleur résultat en 2026.
Toutes les erreurs sont en °C ; les fortes variations concernent uniquement 2026.

| Expérience / variante retenue | MAE 2025 | MAE 2026 | MAE fortes variations |
|---|---:|---:|---:|
| Persistance | 2,176 | 2,039 | 5,683 |
| 01 — XGBoost + mois cyclique | 2,059 | 2,697 | 4,314 |
| 02 — Ridge + observations horaires | 1,996 | 1,988 | 3,799 |
| 03 — XGBoost + voisins à 200 km | 1,967 | 2,321 | 3,685 |
| 04 — Ridge spatial, cible en variation | **1,850** | **1,766** | **3,642** |
| 05 — Ridge, 24 températures puis maximum | 1,924 | 1,922 | 3,705 |

Les comparaisons à modèle et entrées contrôlés montrent plusieurs effets :

- **L'information horaire aide** : XGBoost passe de 2,665 à 2,353 °C de MAE en
  2026 ; Ridge de 2,078 à 1,988 °C. Le gain de Ridge reste incertain dans
  l'intervalle descriptif par blocs. Ces gains ne suffisent pas à résoudre les pics.
- **La cible compte aussi** : avec les mêmes données spatiales, XGBoost passe
  de 2,465 °C en cible directe à 1,718 °C en cible de variation. Ce score 2026
  n'a pas servi à choisir la variante : la validation 2025 retient Ridge spatial.
  La baisse de MAE globale de XGBoost s'accompagne ici d'une hausse de son erreur
  sur fortes variations, de 3,752 à 4,519 °C.
- **Le voisinage a un apport conditionnel** : pour XGBoost en variation,
  l'ajout des données spatiales réduit la MAE de 1,814 à 1,718 °C, mais augmente
  l'erreur sur fortes variations de 4,279 à 4,519 °C. Avec la cible directe,
  les voisins à 200 km ne gagnent que 0,032 °C face au témoin horaire,
  avec un intervalle descriptif incluant zéro.
- **La pondération testée ne gagne pas** : XGBoost spatial en variation passe
  de 1,718 à 1,741 °C, et son erreur sur fortes variations augmente aussi.
- **La décomposition en 24 heures n'apporte pas de gain net robuste** : elle
  réduit légèrement l'erreur de XGBoost, mais dégrade légèrement celle de Ridge.

Le modèle retenu de l'expérience 04 améliore la MAE de 13,4 % face à la
persistance sur 2026, et de 35,9 % sur les fortes variations. Cela soutient
l'intérêt de combiner information supplémentaire et choix de cible, sans prouver
que les fronts météorologiques sont correctement anticipés. Les dates restent
exploratoires et les événements rares sont peu nombreux.

Les valeurs détaillées et les intervalles des comparaisons appariées sont dans
[`summary.json`](experiments/results/summary.json) et
[`paired_comparisons.json`](experiments/results/paired_comparisons.json).
On peut les recalculer sans réentraîner les modèles :

```bash
nix develop --command python scripts/summarize_research.py
```

## Collecte et reproduction

`scripts/heatwave_data.py` et les anciens CSV sont conservés. Le nouveau collecteur
`scripts/fetch_spatial_data.py` utilise ERA5, une sélection de cellule au plus proche
et aucune correction d'altitude. Il télécharge Nantes et huit points cardinaux,
y compris des points en mer à l'ouest, dans `data/research_v1/`. Les cellules
réellement renvoyées doivent être distinctes ; les heures manquantes sont refusées.
Les réponses sont mises en cache par paramètres, permettant de reprendre une collecte.

```bash
nix develop --command python scripts/fetch_spatial_data.py
nix develop --command python -m unittest discover -s tests -v
nix develop --command python scripts/execute_notebooks.py \
  experiments/research/01_calendar_diagnostics.ipynb
```

Exécuter ensuite les notebooks 02 à 05 de la même façon, ou les ouvrir dans
Jupyter avec le noyau **Python (STASC Nix)**. XGBoost est inclus dans l'environnement.
`scripts/create_research_notebooks.py` contient les plans des expériences ; sa
réexécution recrée les notebooks et efface leurs sorties. Elle n'est pas nécessaire
pour relancer un notebook existant.

ERA5 est une réanalyse disponible avec retard : ces expériences ne simulent pas
la disponibilité réelle des données à 00 h. Une confirmation opérationnelle
nécessitera une période non consultée et des prévisions archivées disponibles
à l'heure d'émission. Avec environ mille journées, le passage aux réseaux
neuronaux n'est pas automatique.

Sources : [API historique Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api),
[évaluation temporelle scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).
