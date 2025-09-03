Spectral Unmixing Playground (Python)
====================================

Reproducible playground to synthesize spectral fluorescence datasets and benchmark unmixing algorithms.

Quick start
-----------

1) Install (inside a virtual environment):

- Install dependencies from `pyproject.toml` with your preferred tool.

2) Run an example experiment:

- Use the CLI module with a YAML config, e.g. `config/examples/overdetermined.yaml`.
 - Example Python invocation:
   - `python -m spectral_playground.experiments.run --cfg config/examples/overdetermined.yaml --seed 123`

Testing
-------

- Run your Python test runner on the `tests/` directory.

GUI
---

- Launch the GUI (multiple ways):
  - `python -m spectral_playground.gui.main_gui` (preferred)
  - `python -m spectral_playground.gui.app` (wrapper)
  - `python spectral_playground/gui/app.py` (direct script)
  - Configure grid/channels/fluors/spatial/noise, select methods, then click Run.
  - Features: Dynamic fluorophore editor, object placement layers, continuous spectral plots, noise toggle

Notes
-----

- Shapes: `Y` is `(L, P)`, `M` is `(L, K)`, `A` is `(K, P)`.
- Randomness uses `numpy.random.Generator(PCG64)` seeded from config for reproducibility.


