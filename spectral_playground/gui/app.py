from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore
from spectral_playground.core.spatial import FieldSpec, AbundanceField
from spectral_playground.core.background import BackgroundModel
from spectral_playground.core.noise import NoiseModel
from spectral_playground.core.simulate import ForwardConfig, ForwardModel
from spectral_playground.eval.metrics import rmse, sam
from spectral_playground.experiments.registry import make_unmixer

# Import the improved version
from .app_improved import PlaygroundGUI as ImprovedGUI


class PlaygroundGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Spectral Unmixing Playground")
        # State
        self.current_Y = None
        self.current_A = None
        self.current_B = None
        self.current_M = None
        self.current_field = None
        self.current_spectral = None
        self.channel_vars = []
        self.fluor_vars = []
        self.fluor_vars = []
        self._build_widgets()

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.controls = ttk.Frame(self)
        self.controls.grid(row=0, column=0, sticky="nsw", padx=8, pady=8)

        # Settings header with collapse toggle
        header = ttk.Frame(self.controls)
        header.grid(row=0, column=0, columnspan=4, sticky="w")
        ttk.Label(header, text="Settings").grid(row=0, column=0, sticky="w")
        ttk.Button(header, text="Hide", command=self._toggle_settings).grid(row=0, column=1, padx=(8, 0))

        # Grid params
        ttk.Label(self.controls, text="Grid (nm): start, stop, step").grid(row=1, column=0, sticky="w")
        self.grid_start = tk.DoubleVar(value=450.0)
        self.grid_stop = tk.DoubleVar(value=700.0)
        self.grid_step = tk.DoubleVar(value=1.0)
        ttk.Entry(self.controls, textvariable=self.grid_start, width=8).grid(row=2, column=0, sticky="w")
        ttk.Entry(self.controls, textvariable=self.grid_stop, width=8).grid(row=2, column=1, sticky="w")
        ttk.Entry(self.controls, textvariable=self.grid_step, width=8).grid(row=2, column=2, sticky="w")

        # Channels and Fluors
        ttk.Label(self.controls, text="Channels L / bandwidth nm").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.num_channels = tk.IntVar(value=4)
        self.bandwidth = tk.DoubleVar(value=30.0)
        ttk.Entry(self.controls, textvariable=self.num_channels, width=6).grid(row=4, column=0, sticky="w")
        ttk.Entry(self.controls, textvariable=self.bandwidth, width=8).grid(row=4, column=1, sticky="w")

        ttk.Label(self.controls, text="Fluorophores K").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.num_fluors = tk.IntVar(value=3)
        ttk.Entry(self.controls, textvariable=self.num_fluors, width=6).grid(row=6, column=0, sticky="w")
        ttk.Label(self.controls, text="Fluor models (comma-separated)").grid(row=5, column=1, columnspan=3, sticky="w")
        self.fluor_models_str = tk.StringVar(value="gaussian,skewnorm,lognormal")
        ttk.Entry(self.controls, textvariable=self.fluor_models_str, width=36).grid(row=6, column=1, columnspan=3, sticky="w")

        # Spatial
        ttk.Label(self.controls, text="Field H x W (px) / pixel nm").grid(row=7, column=0, sticky="w", pady=(8, 0))
        self.H = tk.IntVar(value=128)
        self.W = tk.IntVar(value=128)
        self.pixel_nm = tk.DoubleVar(value=100.0)
        ttk.Entry(self.controls, textvariable=self.H, width=6).grid(row=8, column=0, sticky="w")
        ttk.Entry(self.controls, textvariable=self.W, width=6).grid(row=8, column=1, sticky="w")
        ttk.Entry(self.controls, textvariable=self.pixel_nm, width=8).grid(row=8, column=2, sticky="w")

        ttk.Label(self.controls, text="Dot density (/100x100um^2)").grid(row=9, column=0, sticky="w", pady=(8, 0))
        self.density = tk.DoubleVar(value=50.0)
        ttk.Entry(self.controls, textvariable=self.density, width=8).grid(row=10, column=0, sticky="w")

        # Noise
        ttk.Label(self.controls, text="Noise: gain / read_sigma / dark_rate").grid(row=11, column=0, sticky="w", pady=(8, 0))
        self.gain = tk.DoubleVar(value=1.0)
        self.read_sigma = tk.DoubleVar(value=1.0)
        self.dark_rate = tk.DoubleVar(value=0.0)
        ttk.Entry(self.controls, textvariable=self.gain, width=8).grid(row=12, column=0, sticky="w")
        ttk.Entry(self.controls, textvariable=self.read_sigma, width=8).grid(row=12, column=1, sticky="w")
        ttk.Entry(self.controls, textvariable=self.dark_rate, width=8).grid(row=12, column=2, sticky="w")

        # Methods
        ttk.Label(self.controls, text="Methods").grid(row=13, column=0, sticky="w", pady=(8, 0))
        self.use_nnls = tk.BooleanVar(value=True)
        self.use_lasso = tk.BooleanVar(value=False)
        self.use_nmf = tk.BooleanVar(value=False)
        self.use_em = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.controls, text="NNLS", variable=self.use_nnls).grid(row=14, column=0, sticky="w")
        ttk.Checkbutton(self.controls, text="LASSO", variable=self.use_lasso).grid(row=14, column=1, sticky="w")
        ttk.Checkbutton(self.controls, text="NMF", variable=self.use_nmf).grid(row=14, column=2, sticky="w")
        ttk.Checkbutton(self.controls, text="EM (Poisson)", variable=self.use_em).grid(row=14, column=3, sticky="w")
        ttk.Label(self.controls, text="LASSO alpha").grid(row=15, column=0, sticky="w")
        self.lasso_alpha = tk.DoubleVar(value=0.01)
        ttk.Entry(self.controls, textvariable=self.lasso_alpha, width=8).grid(row=16, column=0, sticky="w")
        ttk.Label(self.controls, text="NMF iters").grid(row=15, column=1, sticky="w")
        self.nmf_iters = tk.IntVar(value=200)
        ttk.Entry(self.controls, textvariable=self.nmf_iters, width=8).grid(row=16, column=1, sticky="w")
        ttk.Label(self.controls, text="EM iters").grid(row=15, column=2, sticky="w")
        self.em_iters = tk.IntVar(value=150)
        ttk.Entry(self.controls, textvariable=self.em_iters, width=8).grid(row=16, column=2, sticky="w")

        # Seed and run
        ttk.Label(self.controls, text="Seed").grid(row=17, column=0, sticky="w", pady=(8, 0))
        self.seed = tk.IntVar(value=123)
        ttk.Entry(self.controls, textvariable=self.seed, width=8).grid(row=18, column=0, sticky="w")
        btnbar = ttk.Frame(self.controls)
        btnbar.grid(row=19, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(btnbar, text="Generate Data", command=self.on_generate).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btnbar, text="Run Unmix", command=self.on_unmix).grid(row=0, column=1)

        # Channel toggles (populated after data is generated)
        ttk.Label(self.controls, text="Active channels (visualization)").grid(row=20, column=0, sticky="w", pady=(12, 0))
        self.channel_checks_frame = ttk.Frame(self.controls)
        self.channel_checks_frame.grid(row=21, column=0, columnspan=4, sticky="w")
        btns = ttk.Frame(self.controls)
        btns.grid(row=22, column=0, columnspan=4, sticky="w")
        ttk.Button(btns, text="All", command=self._select_all_channels).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(btns, text="None", command=self._select_no_channels).grid(row=0, column=1, padx=(0, 4))

        # Fluor toggles (populated after data is generated)
        ttk.Label(self.controls, text="Visible fluorophores (GT overlay)").grid(row=23, column=0, sticky="w", pady=(12, 0))
        self.fluor_checks_frame = ttk.Frame(self.controls)
        self.fluor_checks_frame.grid(row=24, column=0, columnspan=4, sticky="w")
        fbtns = ttk.Frame(self.controls)
        fbtns.grid(row=25, column=0, columnspan=4, sticky="w")
        ttk.Button(fbtns, text="All", command=self._select_all_fluors).grid(row=0, column=0, padx=(0, 4))
        ttk.Button(fbtns, text="None", command=self._select_no_fluors).grid(row=0, column=1, padx=(0, 4))

        # Right notebook with two tabs: Data and Unmix
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        self.tab_data = ttk.Frame(self.notebook)
        self.tab_unmix = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="1) Data")
        self.notebook.add(self.tab_unmix, text="2) Unmix")

        # Data tab figure
        self.data_figure = Figure(figsize=(7, 4), dpi=100)
        self.data_canvas = FigureCanvasTkAgg(self.data_figure, master=self.tab_data)
        self.data_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Unmix tab figure and output
        self.unmix_figure = Figure(figsize=(7, 4), dpi=100)
        self.unmix_canvas = FigureCanvasTkAgg(self.unmix_figure, master=self.tab_unmix)
        self.unmix_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(self, height=10)
        self.output.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)

    def _log(self, msg: str) -> None:
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)

    def on_generate(self) -> None:
        try:
            rng = np.random.default_rng(int(self.seed.get()))
            start = float(self.grid_start.get())
            stop = float(self.grid_stop.get())
            step = float(self.grid_step.get())
            lambdas = np.arange(start, stop + 1e-9, step, dtype=np.float32)

            L = int(self.num_channels.get())
            bw = float(self.bandwidth.get())
            centers = np.linspace(start + 0.5 * bw, stop - 0.5 * bw, L)
            channels = [Channel(name=f"C{idx+1}", center_nm=float(c), bandwidth_nm=bw) for idx, c in enumerate(centers)]

            K = int(self.num_fluors.get())
            models = [m.strip().lower() for m in self.fluor_models_str.get().split(",") if m.strip()]
            if len(models) < K:
                models = models + ["gaussian"] * (K - len(models))
            mus = np.linspace(start + 0.1 * (stop - start), stop - 0.1 * (stop - start), K)
            fluors = []
            for k in range(K):
                model = models[k]
                if model == "gaussian":
                    params = {"mu": float(mus[k]), "sigma": 12.0}
                elif model == "skewnorm":
                    params = {"mu": float(mus[k]), "sigma": 12.0, "alpha": 4.0}
                elif model == "lognormal":
                    params = {"mu": np.log(max(mus[k], 1.0)), "sigma": 0.08}
                elif model == "weibull":
                    params = {"k": 2.0, "lam": 20.0, "shift": float(mus[k])}
                else:
                    params = {"mu": float(mus[k]), "sigma": 12.0}
                fluors.append(Fluorophore(name=f"F{k+1}", model=model, params=params, brightness=1.0))

            spectral = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
            M = spectral.build_M()

            H = int(self.H.get())
            W = int(self.W.get())
            px_nm = float(self.pixel_nm.get())
            field = FieldSpec(shape=(H, W), pixel_size_nm=px_nm)
            af = AbundanceField(rng)
            A = af.sample(K=K, field=field, kind="dots", density_per_100x100_um2=float(self.density.get()), spot_profile={"kind": "gaussian", "sigma_px": 1.2})

            bg = BackgroundModel(rng)
            noise = NoiseModel(rng)
            fwd = ForwardModel(spectral=spectral, field=field, bg=bg, noise=noise, cfg=ForwardConfig())
            B = bg.sample(M.shape[0], H, W, kind="constant", level=0.0)
            Y = fwd.synthesize(A, B=B, noise_params={"gain": float(self.gain.get()), "read_sigma": float(self.read_sigma.get()), "dark_rate": float(self.dark_rate.get())})

            # Save state
            self.current_Y = Y
            self.current_A = A
            self.current_B = B
            self.current_M = M
            self.current_field = field
            self.current_spectral = spectral

            # Build toggles and render data view
            self._populate_channel_toggles(L)
            self._populate_fluor_toggles(K)
            self._render_data_view()

            self._log(f"Generated data: L={L}, K={K}, HxW={H}x{W}")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_unmix(self) -> None:
        try:
            if self.current_Y is None:
                messagebox.showinfo("Info", "Generate data first.")
                return
            Y = self.current_Y
            M = self.current_M
            A = self.current_A
            B = self.current_B
            H, W = self.current_field.shape

            methods = []
            if self.use_nnls.get():
                methods.append({"method": "nnls"})
            if self.use_lasso.get():
                methods.append({"method": "lasso", "alpha": float(self.lasso_alpha.get())})
            if self.use_nmf.get():
                methods.append({"method": "nmf", "K": M.shape[1], "n_iter": int(self.nmf_iters.get())})
            if self.use_em.get():
                methods.append({"method": "em_poisson", "n_iter": int(self.em_iters.get())})

            self.output.delete("1.0", tk.END)
            self._log(f"Unmixing with {len(methods)} method(s)")
            self.unmix_figure.clear()
            ax_grid = self.unmix_figure.add_subplot(1, 2, 1)
            im = ax_grid.imshow(M, aspect="auto", origin="lower")
            ax_grid.set_title("M (LxK)")
            self.unmix_figure.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.04)
            ax_maps = self.unmix_figure.add_subplot(1, 2, 2)
            ax_maps.set_axis_off()
            ax_maps.set_title("A maps (first up to 3)")

            for spec in methods:
                un = make_unmixer(spec)
                known_M = None if (spec.get("method") == "nmf" and spec.get("K") is not None) else M
                un.fit(Y, M=known_M)
                out = un.transform(Y, M=known_M)
                A_hat = out.get("A")
                M_hat = out.get("M", known_M)
                m_rmse = rmse(Y, (M_hat @ A_hat) + B) if (A_hat is not None and M_hat is not None) else None
                a_rmse = rmse(A_hat, A) if A_hat is not None else None
                m_sam = sam(M_hat, M, axis=0) if M_hat is not None else None
                self._log(f"{un.name}: rmse_Y={m_rmse:.4f}  rmse_A={a_rmse:.4f}  sam_M={m_sam:.4f}")

                if A_hat is not None:
                    ax_maps.clear()
                    K_show = min(A_hat.shape[0], 3)
                    for i in range(K_show):
                        sub = self.unmix_figure.add_axes([0.55, 0.1 + 0.28 * (K_show - i - 1), 0.4, 0.25])
                        sub.imshow(A_hat[i].reshape(H, W), cmap="magma")
                        sub.set_title(f"{un.name}: A[{i}]")
                        sub.axis("off")
                self.unmix_canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", str(e))
    def on_run(self) -> None:
        try:
            rng = np.random.default_rng(int(self.seed.get()))
            start = float(self.grid_start.get())
            stop = float(self.grid_stop.get())
            step = float(self.grid_step.get())
            lambdas = np.arange(start, stop + 1e-9, step, dtype=np.float32)

            L = int(self.num_channels.get())
            bw = float(self.bandwidth.get())
            centers = np.linspace(start + 0.5 * bw, stop - 0.5 * bw, L)
            channels = [Channel(name=f"C{idx+1}", center_nm=float(c), bandwidth_nm=bw) for idx, c in enumerate(centers)]

            K = int(self.num_fluors.get())
            # Create K Gaussian-like fluorophores spaced across grid
            mus = np.linspace(start + 0.1 * (stop - start), stop - 0.1 * (stop - start), K)
            fluors = [Fluorophore(name=f"F{k+1}", model="gaussian", params={"mu": float(mu), "sigma": 12.0}, brightness=1.0) for k, mu in enumerate(mus)]

            spectral = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
            M = spectral.build_M()

            H = int(self.H.get())
            W = int(self.W.get())
            px_nm = float(self.pixel_nm.get())
            field = FieldSpec(shape=(H, W), pixel_size_nm=px_nm)
            af = AbundanceField(rng)
            A = af.sample(K=K, field=field, kind="dots", density_per_100x100_um2=float(self.density.get()), spot_profile={"kind": "gaussian", "sigma_px": 1.2})

            bg = BackgroundModel(rng)
            noise = NoiseModel(rng)
            fwd = ForwardModel(spectral=spectral, field=field, bg=bg, noise=noise, cfg=ForwardConfig())
            B = bg.sample(M.shape[0], H, W, kind="constant", level=0.0)
            Y = fwd.synthesize(A, B=B, noise_params={"gain": float(self.gain.get()), "read_sigma": float(self.read_sigma.get()), "dark_rate": float(self.dark_rate.get())})

            # Save state
            self.current_Y = Y
            self.current_A = A
            self.current_B = B
            self.current_M = M
            self.current_field = field
            self.current_spectral = spectral

            # Run selected methods
            methods = []
            if self.use_nnls.get():
                methods.append({"method": "nnls"})
            if self.use_lasso.get():
                methods.append({"method": "lasso", "alpha": float(self.lasso_alpha.get())})
            if self.use_nmf.get():
                methods.append({"method": "nmf", "K": K, "n_iter": int(self.nmf_iters.get())})
            if self.use_em.get():
                methods.append({"method": "em_poisson", "n_iter": int(self.em_iters.get())})

            # Build channel toggles and render data view
            self._populate_channel_toggles(L)
            self._render_data_view()

            # Prepare unmix tab
            self.output.delete("1.0", tk.END)
            self._log(f"Running with L={L}, K={K}, HxW={H}x{W}")
            self.unmix_figure.clear()
            ax_grid = self.unmix_figure.add_subplot(1, 2, 1)
            im = ax_grid.imshow(M, aspect="auto", origin="lower")
            ax_grid.set_title("M (LxK)")
            self.unmix_figure.colorbar(im, ax=ax_grid, fraction=0.046, pad=0.04)
            ax_maps = self.unmix_figure.add_subplot(1, 2, 2)
            ax_maps.set_axis_off()
            ax_maps.set_title("A maps (first up to 3)")

            for spec in methods:
                un = make_unmixer(spec)
                known_M = None if (spec.get("method") == "nmf" and spec.get("K") is not None) else M
                un.fit(Y, M=known_M)
                out = un.transform(Y, M=known_M)
                A_hat = out.get("A")
                M_hat = out.get("M", known_M)
                m_rmse = rmse(Y, (M_hat @ A_hat) + B) if (A_hat is not None and M_hat is not None) else None
                a_rmse = rmse(A_hat, A) if A_hat is not None else None
                m_sam = sam(M_hat, M, axis=0) if M_hat is not None else None
                self._log(f"{un.name}: rmse_Y={m_rmse:.4f}  rmse_A={a_rmse:.4f}  sam_M={m_sam:.4f}")

                # Plot first few abundance maps
                if A_hat is not None:
                    ax_maps.clear()
                    K_show = min(A_hat.shape[0], 3)
                    for i in range(K_show):
                        sub = self.unmix_figure.add_axes([0.55, 0.1 + 0.28 * (K_show - i - 1), 0.4, 0.25])
                        sub.imshow(A_hat[i].reshape(H, W), cmap="magma")
                        sub.set_title(f"{un.name}: A[{i}]")
                        sub.axis("off")
                self.unmix_canvas.draw_idle()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _populate_channel_toggles(self, L: int) -> None:
        for w in self.channel_checks_frame.winfo_children():
            w.destroy()
        self.channel_vars = []
        for i in range(L):
            var = tk.BooleanVar(value=True)
            self.channel_vars.append(var)
            ttk.Checkbutton(self.channel_checks_frame, text=f"C{i+1}", variable=var, command=self._render_data_view).grid(row=0, column=i, sticky="w")

    def _select_all_channels(self) -> None:
        for v in self.channel_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_channels(self) -> None:
        for v in self.channel_vars:
            v.set(False)
        self._render_data_view()

    def _populate_fluor_toggles(self, K: int) -> None:
        for w in self.fluor_checks_frame.winfo_children():
            w.destroy()
        self.fluor_vars = []
        for i in range(K):
            var = tk.BooleanVar(value=True)
            self.fluor_vars.append(var)
            ttk.Checkbutton(self.fluor_checks_frame, text=f"F{i+1}", variable=var, command=self._render_data_view).grid(row=0, column=i, sticky="w")

    def _select_all_fluors(self) -> None:
        for v in self.fluor_vars:
            v.set(True)
        self._render_data_view()

    def _select_no_fluors(self) -> None:
        for v in self.fluor_vars:
            v.set(False)
        self._render_data_view()

    def _render_data_view(self) -> None:
        if self.current_Y is None or self.current_field is None or self.current_spectral is None:
            return
        H, W = self.current_field.shape
        L, P = self.current_Y.shape
        active = [i for i, v in enumerate(self.channel_vars) if v.get()]
        # Build RGB composite
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        eps = 1e-6
        for i in active:
            img = self.current_Y[i].reshape(H, W)
            img = img / (np.percentile(img, 99.0) + eps)
            img = np.clip(img, 0.0, 1.0)
            nm = self.current_spectral.channels[i].center_nm
            color = np.array(self._wavelength_to_rgb_nm(nm), dtype=np.float32)
            rgb += img[..., None] * color[None, None, :]
        rgb = np.clip(rgb, 0.0, 1.0)

        # Plot image and spectral bands
        self.data_figure.clear()
        ax_img = self.data_figure.add_subplot(2, 1, 1)
        ax_img.imshow(rgb)
        ax_img.set_title("Composite of selected channels")
        ax_img.axis("off")

        ax_spec = self.data_figure.add_subplot(2, 1, 2)
        start = float(self.grid_start.get())
        stop = float(self.grid_stop.get())
        ax_spec.set_xlim(start, stop)
        ax_spec.set_ylim(0, 1)
        ax_spec.set_yticks([])
        ax_spec.set_xlabel("Wavelength (nm)")
        for idx, ch in enumerate(self.current_spectral.channels):
            half = 0.5 * ch.bandwidth_nm
            low, high = ch.center_nm - half, ch.center_nm + half
            active_flag = idx in active
            face = self._wavelength_to_rgb_nm(ch.center_nm) if active_flag else (0.6, 0.6, 0.6)
            ax_spec.axvspan(low, high, ymin=0.1, ymax=0.9, facecolor=face, alpha=0.4 if active_flag else 0.2, edgecolor='k')
            ax_spec.text(ch.center_nm, 0.95, f"C{idx+1}", ha='center', va='top', fontsize=8)
        # Overlay fluor PDFs and total mix
        if self.current_A is not None:
            K = self.current_A.shape[0]
            sum_A = np.sum(self.current_A, axis=1)
            total = np.zeros_like(self.current_spectral.lambdas, dtype=np.float32)
            for k in range(K):
                show = (k < len(self.fluor_vars) and self.fluor_vars[k].get())
                pdf = self.current_spectral._pdf(self.current_spectral.fluors[k])
                total += pdf * float(sum_A[k])
                if show:
                    peak_nm = float(self.current_spectral.lambdas[np.argmax(pdf)])
                    col = self._wavelength_to_rgb_nm(peak_nm)
                    ax_spec.plot(self.current_spectral.lambdas, pdf / (np.max(pdf) + 1e-9), color=col, linewidth=1.5, label=f"F{k+1}")
            if np.max(total) > 0:
                ax_spec.plot(self.current_spectral.lambdas, total / np.max(total), color='k', linewidth=2.0, label='Total (GT)')
            # Channel total intensities as markers
            ch_sums = np.sum(self.current_Y, axis=1)
            ch_sums = ch_sums / (np.max(ch_sums) + 1e-9)
            centers = [ch.center_nm for ch in self.current_spectral.channels]
            ax_spec.plot(centers, ch_sums, 'o', color='k', markersize=4, label='Channel totals')
            ax_spec.legend(loc='upper right', fontsize=8, ncol=2)

        # Show selected GT abundance maps (first up to 3)
        if self.current_A is not None and len(self.fluor_vars) > 0:
            sel_f = [i for i, v in enumerate(self.fluor_vars) if v.get()]
            # Remove previous small axes if any
            for ax in list(self.data_figure.axes)[2:]:
                self.data_figure.delaxes(ax)
            K_show = min(len(sel_f), 3)
            for i in range(K_show):
                sub = self.data_figure.add_axes([0.1 + i * 0.28, 0.02, 0.25, 0.18])
                sub.imshow(self.current_A[sel_f[i]].reshape(H, W), cmap="magma")
                sub.set_title(f"GT A[{sel_f[i]}]")
                sub.axis("off")
        self.data_canvas.draw_idle()

    def _toggle_settings(self) -> None:
        # Collapse/expand left settings panel
        if self.controls.winfo_ismapped():
            self.controls.grid_remove()
        else:
            self.controls.grid()

    @staticmethod
    def _wavelength_to_rgb_nm(nm: float) -> tuple[float, float, float]:
        # Approximate wavelength to RGB (380-780nm)
        w = nm
        if w < 380 or w > 780:
            return (0.0, 0.0, 0.0)
        if w < 440:
            r = -(w - 440) / (440 - 380)
            g = 0.0
            b = 1.0
        elif w < 490:
            r = 0.0
            g = (w - 440) / (490 - 440)
            b = 1.0
        elif w < 510:
            r = 0.0
            g = 1.0
            b = -(w - 510) / (510 - 490)
        elif w < 580:
            r = (w - 510) / (580 - 510)
            g = 1.0
            b = 0.0
        elif w < 645:
            r = 1.0
            g = -(w - 645) / (645 - 580)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0
        # Intensity factor near spectrum edges
        if w < 420:
            f = 0.3 + 0.7 * (w - 380) / (420 - 380)
        elif w > 700:
            f = 0.3 + 0.7 * (780 - w) / (780 - 700)
        else:
            f = 1.0
        return (float(r * f), float(g * f), float(b * f))


def main() -> None:
    # Use the improved GUI by default
    app = ImprovedGUI()
    app.mainloop()


if __name__ == "__main__":
    main()


