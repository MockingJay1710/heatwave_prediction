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
