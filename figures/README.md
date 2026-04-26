# Figure regeneration

These scripts regenerate the main-text and SI figures from the bundled scored data.

## Scripts

| Script | Generates |
|---|---|
| `main_figures.py` | Main-text figures (headline inverted-U, quintile panels, cross-paradigm comparison) |
| `si_figures.py` | SI figures (validation, scorer-checkpoint robustness, cross-content stability, etc.) |

## Run

```bash
mkdir -p figures/output
cd figures
python main_figures.py
python si_figures.py
```

Both scripts write PDFs and PNGs to `figures/output/`. The bundled data files under `../data/` are sufficient for full regeneration; no API calls or scorer inference are required.

## Dependencies

`matplotlib`, `numpy`, `pandas`, `scipy` — all in `requirements.txt`. Optional: `scienceplots` for an alternative style; the scripts fall back to the default PNAS-style rcParams if unavailable.
