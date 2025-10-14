"""Image I/O utilities for saving and loading spectral data."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tifffile

from .dataset import SynthDataset
from ..core.spectra import SpectralSystem, Channel, Fluorophore
from ..core.spatial import FieldSpec


class SpectralImageIO:
    """Handles saving and loading of spectral imaging data."""
    
    @staticmethod
    def save_composite_image(
        rgb_data: np.ndarray, 
        filepath: str, 
        format: str = "PNG",
        dpi: int = 300
    ) -> None:
        """Save RGB composite image as PNG/JPG for viewing.
        
        Args:
            rgb_data: RGB image data (H, W, 3) with values in [0, 1]
            filepath: Output file path
            format: Image format ("PNG" or "JPEG")
            dpi: Resolution for saved image
        """
        # Convert to 8-bit and ensure proper format
        rgb_uint8 = (np.clip(rgb_data, 0, 1) * 255).astype(np.uint8)
        
        # Create PIL image
        img = Image.fromarray(rgb_uint8, mode='RGB')
        
        # Save with metadata
        if format.upper() == "PNG":
            img.save(filepath, format="PNG", dpi=(dpi, dpi), optimize=True)
        elif format.upper() in ["JPEG", "JPG"]:
            img.save(filepath, format="JPEG", dpi=(dpi, dpi), quality=95, optimize=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def save_multichannel_tiff(
        Y: np.ndarray,
        field_shape: Tuple[int, int],
        channels: list,
        filepath: str,
        bit_depth: int = 16,
        normalization: str = "per_channel",
        pixel_size_nm: Optional[float] = None
    ) -> None:
        """Save multi-channel spectral data as TIFF.
        
        Creates a multi-page TIFF where each page is one detection channel.
        Compatible with ImageJ, FIJI, and other microscopy software.
        
        Args:
            Y: Channel data, shape (L, H*W) or (L, H, W)
            field_shape: Spatial dimensions (H, W)
            channels: List of Channel objects with metadata
            filepath: Output file path (.tif or .tiff)
            bit_depth: Output bit depth (8, 16, or 32)
            normalization: Scaling strategy:
                - "per_channel": Each channel scaled to its own max (default)
                - "global": All channels use same scale
                - "none": Keep raw values (photon counts)
            pixel_size_nm: Physical pixel size in nanometers (for metadata)
        """
        H, W = field_shape
        L = Y.shape[0]
        
        # Reshape if needed
        if Y.ndim == 2:
            Y_shaped = Y.reshape(L, H, W)
        else:
            Y_shaped = Y
        
        # Prepare channel data based on normalization
        if normalization == "none":
            # Keep raw values - use float32
            channel_stack = Y_shaped.astype(np.float32)
            bit_depth = 32  # Force 32-bit for raw data
        else:
            # Normalize data
            if normalization == "global":
                # Global normalization across all channels
                global_max = np.max(Y_shaped)
                if global_max > 0:
                    normalized = Y_shaped / global_max
                else:
                    normalized = Y_shaped
            else:  # per_channel
                # Per-channel normalization
                normalized = np.zeros_like(Y_shaped, dtype=np.float32)
                for i in range(L):
                    channel_max = np.max(Y_shaped[i])
                    if channel_max > 0:
                        normalized[i] = Y_shaped[i] / channel_max
                    else:
                        normalized[i] = Y_shaped[i]
            
            # Convert to appropriate bit depth
            if bit_depth == 8:
                channel_stack = (np.clip(normalized, 0, 1) * 255).astype(np.uint8)
            elif bit_depth == 16:
                channel_stack = (np.clip(normalized, 0, 1) * 65535).astype(np.uint16)
            else:  # 32-bit float
                channel_stack = normalized.astype(np.float32)
        
        # Build metadata
        metadata = {
            'axes': 'CYX',  # Channel, Y, X
            'Channel': {'Name': [ch.name for ch in channels]},
        }
        
        # Add channel-specific metadata
        channel_metadata = {}
        for idx, ch in enumerate(channels):
            channel_metadata[f'Channel_{idx}'] = {
                'Name': ch.name,
                'CenterWavelength_nm': ch.center_nm,
                'Bandwidth_nm': ch.bandwidth_nm,
            }
        metadata['ChannelInfo'] = channel_metadata
        
        # Add physical pixel size if available
        if pixel_size_nm is not None:
            # Convert nm to micrometers for standard microscopy metadata
            pixel_size_um = pixel_size_nm / 1000.0
            metadata['PhysicalSizeX'] = pixel_size_um
            metadata['PhysicalSizeY'] = pixel_size_um
            metadata['PhysicalSizeXUnit'] = 'µm'
            metadata['PhysicalSizeYUnit'] = 'µm'
        
        # Save as OME-TIFF with metadata
        tifffile.imwrite(
            filepath,
            channel_stack,
            photometric='minisblack',
            metadata=metadata,
            compression='deflate',
            compressionargs={'level': 6}
        )
    
    @staticmethod
    def save_full_dataset(
        dataset: SynthDataset,
        spectral_system: SpectralSystem,
        field_spec: FieldSpec,
        filepath: str,
        include_metadata: bool = True
    ) -> None:
        """Save complete spectral dataset with all data and metadata.
        
        Args:
            dataset: Complete synthetic dataset
            spectral_system: Spectral system configuration
            field_spec: Spatial field specification
            filepath: Output NPZ file path
            include_metadata: Whether to include system metadata
        """
        # Prepare data dictionary
        save_data = {
            'Y': dataset.Y,
            'A': dataset.A,
            'B': dataset.B,
            'M': dataset.M,
            'field_shape': field_spec.shape,
            'pixel_size_nm': field_spec.pixel_size_nm,
        }
        
        # Add optional data
        if dataset.S is not None:
            save_data['S'] = dataset.S
        
        # Add metadata if requested
        if include_metadata:
            metadata = {
                'wavelengths': spectral_system.lambdas,
                'channels': [
                    {
                        'name': ch.name,
                        'center_nm': ch.center_nm,
                        'bandwidth_nm': ch.bandwidth_nm
                    } for ch in spectral_system.channels
                ],
                'fluorophores': [
                    {
                        'name': fl.name,
                        'model': fl.model,
                        'brightness': fl.brightness,
                        'params': fl.params
                    } for fl in spectral_system.fluors
                ],
                'format_version': '1.0',
                'description': 'Spectral visualization playground dataset',
                'user_metadata': dataset.meta  # Preserve user metadata (objects, etc.)
            }
            save_data['metadata'] = metadata
        
        # Save as compressed NPZ
        np.savez_compressed(filepath, **save_data)
    
    @staticmethod
    def load_full_dataset(filepath: str) -> Tuple[SynthDataset, SpectralSystem, FieldSpec]:
        """Load complete spectral dataset from NPZ file.
        
        Args:
            filepath: Input NPZ file path
            
        Returns:
            Tuple of (dataset, spectral_system, field_spec)
        """
        data = np.load(filepath, allow_pickle=True)
        
        # Extract core data
        Y = data['Y']
        A = data['A']
        B = data['B']
        M = data['M']
        field_shape = tuple(data['field_shape'])
        pixel_size_nm = float(data['pixel_size_nm'])
        
        # Extract optional data
        S = data.get('S', None)
        
        # Extract user metadata if available (will be populated from 'metadata' dict below if it exists)
        user_meta = {}
        
        # Create dataset (meta will be updated below if metadata exists)
        dataset = SynthDataset(
            Y=Y, A=A, B=B, M=M, S=S,
            meta=user_meta
        )
        
        # Reconstruct spectral system and field if metadata exists
        if 'metadata' in data:
            metadata = data['metadata'].item()  # Convert numpy scalar to dict
            
            # Restore user metadata (objects, seed, etc.)
            if 'user_metadata' in metadata:
                dataset.meta = dict(metadata['user_metadata'])
            
            # Reconstruct wavelengths
            lambdas = metadata['wavelengths']
            
            # Reconstruct channels
            channels = []
            for ch_data in metadata['channels']:
                channels.append(Channel(
                    name=ch_data['name'],
                    center_nm=ch_data['center_nm'],
                    bandwidth_nm=ch_data['bandwidth_nm']
                ))
            
            # Reconstruct fluorophores
            fluorophores = []
            for fl_data in metadata['fluorophores']:
                fluorophores.append(Fluorophore(
                    name=fl_data['name'],
                    model=fl_data['model'],
                    brightness=fl_data['brightness'],
                    params=fl_data['params']
                ))
            
            spectral_system = SpectralSystem(
                lambdas=lambdas,
                channels=channels,
                fluors=fluorophores
            )
        else:
            # Create minimal spectral system if no metadata
            L = Y.shape[0]
            lambdas = np.linspace(400, 700, 100)
            channels = [Channel(f"C{i+1}", 500 + i*50, 50) for i in range(L)]
            K = A.shape[0]
            fluorophores = [Fluorophore(f"F{i+1}", "gaussian", 1.0, {"mean": 500 + i*100, "std": 30}) for i in range(K)]
            
            spectral_system = SpectralSystem(
                lambdas=lambdas,
                channels=channels,
                fluors=fluorophores
            )
        
        # Create field spec
        field_spec = FieldSpec(
            shape=field_shape,
            pixel_size_nm=pixel_size_nm
        )
        
        return dataset, spectral_system, field_spec
    
    @staticmethod
    def save_plot_as_image(
        figure: Figure,
        filepath: str,
        format: str = "PNG",
        dpi: int = 300,
        bbox_inches: str = "tight"
    ) -> None:
        """Save matplotlib figure as image file.
        
        Args:
            figure: Matplotlib figure to save
            filepath: Output file path
            format: Image format ("PNG", "JPEG", "SVG", "PDF")
            dpi: Resolution for raster formats
            bbox_inches: Bounding box setting
        """
        figure.savefig(
            filepath,
            format=format.lower(),
            dpi=dpi,
            bbox_inches=bbox_inches,
            facecolor='white',
            edgecolor='none'
        )
    
    @staticmethod
    def export_abundance_maps(
        A: np.ndarray,
        field_shape: Tuple[int, int],
        output_dir: str,
        prefix: str = "abundance",
        format: str = "PNG",
        colormap: str = "magma"
    ) -> None:
        """Export individual abundance maps as separate image files.
        
        Args:
            A: Abundance data (K, H*W)
            field_shape: Spatial dimensions (H, W)
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format
            colormap: Matplotlib colormap name
        """
        os.makedirs(output_dir, exist_ok=True)
        
        K = A.shape[0]
        H, W = field_shape
        
        for k in range(K):
            abundance_map = A[k].reshape(H, W)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
            im = ax.imshow(abundance_map, cmap=colormap)
            ax.set_title(f"Fluorophore {k+1} Abundance")
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Save
            filepath = os.path.join(output_dir, f"{prefix}_F{k+1}.{format.lower()}")
            fig.savefig(filepath, format=format.lower(), dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    @staticmethod
    def get_supported_formats() -> Dict[str, Dict[str, Any]]:
        """Get information about supported file formats.
        
        Returns:
            Dictionary with format information
        """
        return {
            'composite_image': {
                'formats': ['PNG', 'JPEG'],
                'extensions': ['.png', '.jpg', '.jpeg'],
                'description': 'RGB composite images for viewing'
            },
            'full_dataset': {
                'formats': ['NPZ'],
                'extensions': ['.npz'],
                'description': 'Complete spectral data with metadata'
            },
            'plots': {
                'formats': ['PNG', 'JPEG', 'SVG', 'PDF'],
                'extensions': ['.png', '.jpg', '.jpeg', '.svg', '.pdf'],
                'description': 'Matplotlib plots and figures'
            }
        }
    
    @staticmethod
    def validate_file_format(filepath: str, expected_type: str) -> bool:
        """Validate file format matches expected type.
        
        Args:
            filepath: File path to check
            expected_type: Expected format type ('composite_image', 'full_dataset', 'plots')
            
        Returns:
            True if format is valid
        """
        formats_info = SpectralImageIO.get_supported_formats()
        
        if expected_type not in formats_info:
            return False
        
        ext = Path(filepath).suffix.lower()
        return ext in formats_info[expected_type]['extensions']
