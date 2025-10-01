from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Iterable, Optional, List

import os
import numpy as np

from spectral_playground.core.spectra import SpectralSystem, Channel, Fluorophore
from spectral_playground.core.spatial import FieldSpec, AbundanceField
from spectral_playground.core.background import BackgroundModel
from spectral_playground.core.simulate import ForwardConfig, ForwardModel
from spectral_playground.data.image_io import SpectralImageIO
from spectral_playground.data.dataset import SynthDataset

from .state import PlaygroundData


@dataclass
class GenerationConfig:
    seed: int
    grid: Dict[str, float]
    channels: List[Dict[str, Any]]  # Changed from Dict to List
    dimensions: Dict[str, float]


def generate_dataset(
    cfg: GenerationConfig,
    fluorophores: Iterable[Fluorophore],
    objects: Iterable[Dict[str, Any]],
) -> PlaygroundData:
    """Synthesize a dataset based on the GUI configuration."""

    rng = np.random.default_rng(cfg.seed)

    lambdas = np.arange(
        cfg.grid['start'],
        cfg.grid['stop'] + 1e-9,
        cfg.grid['step'],
        dtype=np.float32,
    )

    # Create channels directly from configuration
    channels = [
        Channel(
            name=ch_cfg['name'],
            center_nm=float(ch_cfg['center_nm']),
            bandwidth_nm=float(ch_cfg['bandwidth_nm'])
        )
        for ch_cfg in cfg.channels
    ]

    fluors = list(fluorophores)
    spectral = SpectralSystem(lambdas=lambdas, channels=channels, fluors=fluors)
    M = spectral.build_M()

    H = int(cfg.dimensions['H'])
    W = int(cfg.dimensions['W'])
    pixel_nm = float(cfg.dimensions['pixel_nm'])
    field = FieldSpec(shape=(H, W), pixel_size_nm=pixel_nm)

    af = AbundanceField(rng)
    obj_list = list(objects)
    generated_objects = []
    
    if obj_list:
        A, generated_objects = af.build_from_objects(K=len(fluors), field=field, objects=obj_list, track_objects=True)
    else:
        A = af.sample(K=len(fluors), field=field, kind='uniform')

    bg = BackgroundModel(rng)
    forward = ForwardModel(
        spectral=spectral,
        field=field,
        bg=bg,
        cfg=ForwardConfig(),
    )

    B = bg.sample(M.shape[0], H, W, kind='constant', level=0.0)

    Y = forward.synthesize(A, B=B)

    data = PlaygroundData(
        Y=Y,
        A=A,
        B=B,
        M=M,
        spectral=spectral,
        field=field,
        metadata={'seed': cfg.seed, 'objects': generated_objects},
    )
    return data


def load_dataset(path: str) -> PlaygroundData:
    dataset, spectral, field = SpectralImageIO.load_full_dataset(path)
    data = PlaygroundData(
        Y=dataset.Y,
        A=dataset.A,
        B=dataset.B,
        M=dataset.M,
        spectral=spectral,
        field=field,
        metadata=dict(dataset.meta) if dataset.meta else {},
    )
    return data


def save_dataset(path: str, data: PlaygroundData) -> None:
    if not data.has_data:
        raise ValueError('No dataset available to save.')

    dataset = SynthDataset(
        Y=data.Y,
        A=data.A,
        B=data.B,
        M=data.M,
        S=None,
        meta=data.metadata,
    )
    SpectralImageIO.save_full_dataset(dataset, data.spectral, data.field, path)


def save_composite(path: str, rgb: np.ndarray, format: str = 'PNG') -> None:
    SpectralImageIO.save_composite_image(rgb, path, format)


def save_plots(
    output_dir: str,
    rgb: np.ndarray,
    spectral_figure,
    abundance_figure,
    A: Optional[np.ndarray],
    field_shape: Optional[tuple[int, int]],
    prefix: str = 'abundance',
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    SpectralImageIO.save_composite_image(rgb, os.path.join(output_dir, 'composite_image.png'), 'PNG')
    SpectralImageIO.save_plot_as_image(spectral_figure, os.path.join(output_dir, 'spectral_profiles.png'))
    SpectralImageIO.save_plot_as_image(abundance_figure, os.path.join(output_dir, 'abundance_maps.png'))
    if A is not None and field_shape is not None:
        SpectralImageIO.export_abundance_maps(A, field_shape, output_dir, prefix)
