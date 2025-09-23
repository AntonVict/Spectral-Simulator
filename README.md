Spectral Visualization Playground (Python)
=============================================

Interactive playground for synthesising and exploring multispectral fluorescence datasets. The focus is on rapid iteration of spectral configurations and intuitive visual analysis, without bundled unmixing algorithms.

Features
--------

- **Spectral Simulation**: Configure wavelength grids, detection channels, fluorophore models, and noise to generate synthetic datasets in seconds.
- **Rich Visualization**: Inspect composite images, per-channel responses, spectral profiles, and abundance maps with responsive matplotlib panels.
- **Flexible Object Layouts**: Place objects (dots, circles, boxes, Gaussian blobs) to craft spatial distributions for each fluorophore.
- **Data Management**: Save and load datasets (`.npz`), export publication-ready plots, and keep assets organised under `saved_data/`.

Quick Start
-----------

```bash
python -m spectral_playground.gui.main_gui
```

Installation
------------

```bash
pip install -e .
```

Workflow
--------

1. Configure the spectral system (wavelength grid, channels, fluorophores).
2. Design spatial layouts with the object layers sidebar.
3. Generate data to view composite imagery and spectra instantly.
4. Save datasets or export plots for downstream analysis.

File Organization
-----------------

Generated assets live under `saved_data/`:

- `datasets/` — Full spectral datasets (`.npz`).
- `images/` — Composite renderings (`.png`, `.jpg`).
- `plots/` — Spectral and abundance figures.
- `exports/` — Batch exports created via the GUI.

