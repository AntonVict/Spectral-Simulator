Spectral Unmixing Playground (Python)
====================================

Reproducible playground to synthesize spectral fluorescence datasets and benchmark unmixing algorithms with an interactive GUI.

Features
--------

🔬 **Spectral Simulation**
- Multiple fluorophore models: Gaussian, log-normal, skew-normal, Weibull, mixtures
- Realistic noise models: Poisson, Gaussian, Poisson-Gaussian, multiplicative gain
- Configurable detection channels with wavelength-dependent responses

🎛️ **Interactive GUI**
- Dynamic fluorophore editor with real-time parameter adjustment
- Object placement system: dots, circles, boxes, Gaussian blobs on spatial regions
- Advanced image navigation: zoom, pan, scroll with mouse and keyboard
- Live spectral profile visualization with continuous curves

💾 **Data Management**
- Complete save/load system preserving all spectral data and metadata
- Export composite images (PNG/JPG) for presentations and publications
- Organized directory structure: `saved_data/{datasets,images,plots,exports}/`
- One-click access to save folder from GUI

🧮 **Algorithm Testing**
- Multiple unmixing methods: NNLS, LASSO, NMF, EM-Poisson
- Performance metrics: RMSE, SAM for spectral and abundance recovery
- Support for both overdetermined and underdetermined scenarios

Quick Start
-----------

**GUI (Recommended):**
```bash
python -m spectral_playground.gui.main_gui
```

**CLI Experiments:**
```bash
python -m spectral_playground.experiments.run --cfg config/examples/overdetermined.yaml --seed 123
```

**Installation:**
```bash
# In your virtual environment
pip install -e .  # Install from pyproject.toml
```

Workflow
--------

1. **Configure System**: Set wavelength grid, detection channels, fluorophore properties
2. **Design Spatial Layout**: Place objects with different fluorophore distributions  
3. **Generate Data**: Synthesize realistic spectral measurements with optional noise
4. **Visualize Results**: Explore composite images and spectral profiles interactively
5. **Test Algorithms**: Apply unmixing methods and compare performance
6. **Save/Load Work**: Preserve complete analysis state for reproducible research

File Organization
-----------------

Generated data is automatically organized in `saved_data/`:
- `datasets/` - Complete spectral datasets (.npz) with full metadata
- `images/` - Composite fluorescence images (.png, .jpg)  
- `plots/` - Spectral profiles and abundance maps
- `exports/` - Bulk analysis results and visualizations

Technical Details
-----------------

- **Data shapes**: `Y` (L×P), `M` (L×K), `A` (K×P) where L=channels, K=fluorophores, P=pixels
- **Reproducibility**: Seeded NumPy random generators for consistent results
- **Performance**: Optimized NumPy/SciPy operations with optional Numba acceleration


