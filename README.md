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

## Experiments

- `experiments/heatwave_linear_regression.ipynb`: corrected linear baselines,
  Random Forest and histogram gradient boosting, evaluated on identical test dates.
  Scaling is fitted inside each linear pipeline on training data only;
  persistence and target temperatures stay in degrees Celsius. Feature ablation
  uses an internal validation period. Simple linear plots are saved in
  `experiments/plots_linear/`; enriched linear plots remain in `experiments/plots/`.
- `experiments/heatwave_xgboost_calendar.ipynb`: XGBoost with weather and past
  observations, compared with cyclic month encoding and meteorological seasons.
  All variants share a chronological 60/20/20 train/validation/test split.
  Early stopping and variant selection use validation only; final models are
  refitted before test evaluation. Figures are in `experiments/plots_xgboost_calendar/`.
  Geographic neighbors are reserved for the next experiment.

Both notebooks use the same preparation functions in `scripts/experiment_utils.py`:
targets match the exact next calendar day, historical features match exact past
days, and ambiguous duplicate or multiple-location data are rejected. Training
targets are strictly earlier than the next period's first predictor date.
The CSV must contain the single Nantes ERA5 cell; it is not modified by training.

XGBoost is included in the pinned Nix environment. After changing `flake.nix`,
restart `nix develop` and the notebook kernel **Python (STASC Nix)**.
To rerun both notebooks and the date checks from the project root:

```bash
nix develop --command python -m unittest discover -s tests -v
nix develop --command python scripts/execute_notebooks.py \
  experiments/heatwave_linear_regression.ipynb \
  experiments/heatwave_xgboost_calendar.ipynb
```

The experiments predict next-day Tmax after the end of the observation day,
using retrospective reanalysis data. They do not yet evaluate heatwave events
or operational data availability. Previously inspected test dates provide
exploratory evidence; confirmation needs a future period not used during development.
