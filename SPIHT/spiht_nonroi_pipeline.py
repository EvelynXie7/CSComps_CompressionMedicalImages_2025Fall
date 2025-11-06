"""
spiht_nonroi_pipeline.py

Uniform SPIHT compression pipeline for KiTS19 and BraTS2020 datasets.

Author: Evelyn Xie
"""

import os
import sys
import pathlib
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path setup
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dwt import *
from SPIHT_encoder import *
from SPIHT_decoder import func_MySPIHT_Dec
from kits19.starter_code.visualize import DEFAULT_HU_MIN, DEFAULT_HU_MAX


# Helper Utilities


def find_nifti(base_path):
    """Return .nii.gz or .nii path if exists."""
    if os.path.exists(base_path + ".nii.gz"):
        return base_path + ".nii.gz"
    elif os.path.exists(base_path + ".nii"):
        return base_path + ".nii"
    return None


def _decode_bitstream_to_spatial(bitstream, level, pad_hw):
    """SPIHT decode → inverse DWT → unpad."""
    m_wave = func_MySPIHT_Dec(bitstream)
    rec = decodeDWT(m_wave, level)
    rec = unpad(rec, tuple(pad_hw))
    return rec.astype(np.float32)


# Encoding

def encode_full(img, output_path, level=3, compression_ratio=10):
    """Encode entire image uniformly with SPIHT."""
    k = 2 ** level
    img, pad_hw = pad_to_multiple(img, k)
    img_dwt = runDWT(img, level)
    original_bits = img.size * 8
    max_bits = int(original_bits / compression_ratio)

    bitstream = func_MySPIHT_Enc(img_dwt, max_bits=max_bits, level=level)

    np.savez_compressed(
        output_path,
        bitstream=bitstream,
        shape=img.shape,
        level=level,
        compression_ratio=compression_ratio,
        bits_used=len(bitstream),
        pad_hw=pad_hw
    )


def reconstruct_slice(case_output_dir, slice_idx):
    """Decode one slice from compressed file."""
    npz_path = os.path.join(case_output_dir, f"slice_{slice_idx:03d}.npz")
    meta_path = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")

    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Missing {npz_path} or {meta_path}")

    data = np.load(npz_path)
    meta = np.load(meta_path)

    rec_norm = _decode_bitstream_to_spatial(data["bitstream"], int(data["level"]), data["pad_hw"])
    rec_norm = np.clip(rec_norm, 0, 255)

    norm_min = float(meta["norm_min"])
    norm_max = float(meta["norm_max"])
    rec_orig = (rec_norm / 255.0) * (norm_max - norm_min) + norm_min
    return rec_norm.astype(np.float32), rec_orig.astype(np.float32)



# Dataset-Level Pipelines


def process_kits19_dataset(data_dir, compressed_dir, decompressed_dir,
                           max_cases=2, max_slices=5, ratio=10, level=2):
    """Dataset-level processing for KiTS19."""
    os.makedirs(os.path.join(compressed_dir, "SPIHT"), exist_ok=True)
    os.makedirs(os.path.join(decompressed_dir, "SPIHT"), exist_ok=True)

    for i in range(max_cases):
        case_name = f"case_{i:05d}"
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.exists(case_dir):
            continue

        print(f"[Process KiTS19] {case_name}")
        imaging_path = os.path.join(case_dir, "imaging.nii.gz")
        if not os.path.exists(imaging_path):
            print(f"[Skip] Missing imaging for {case_name}")
            continue

        volume = nib.load(imaging_path).get_fdata()
        num_slices = volume.shape[0]
        if max_slices:
            num_slices = min(num_slices, max_slices)

        case_comp_dir = os.path.join(compressed_dir, "SPIHT", case_name)
        case_decomp_dir = os.path.join(decompressed_dir, "SPIHT", case_name)
        os.makedirs(case_comp_dir, exist_ok=True)
        os.makedirs(case_decomp_dir, exist_ok=True)

        for s in range(num_slices):
            slice_data = volume[s, :, :]
            slice_data = np.clip(slice_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
            normalized = ((slice_data - DEFAULT_HU_MIN) / (DEFAULT_HU_MAX - DEFAULT_HU_MIN) * 255).astype(np.float32)

            output_npz = os.path.join(case_comp_dir, f"slice_{s:03d}.npz")
            encode_full(normalized, output_npz, level, ratio)
            np.savez(os.path.join(case_comp_dir, f"slice_{s:03d}_meta.npz"),
                     norm_min=float(DEFAULT_HU_MIN), norm_max=float(DEFAULT_HU_MAX))

            rec_norm, _ = reconstruct_slice(case_comp_dir, s)
            plt.imsave(os.path.join(case_decomp_dir, f"slice_{s:03d}.png"),
                       rec_norm, cmap="gray", vmin=0, vmax=255)

        print(f"[Done] {case_name}: {num_slices} slices.")


def process_brats_dataset(data_dir, compressed_dir, decompressed_dir,
                          max_cases=2, max_slices=5, ratio=10, level=2, modality="t1ce"):
    """Dataset-level processing for BraTS2020."""
    os.makedirs(os.path.join(compressed_dir, "SPIHT"), exist_ok=True)
    os.makedirs(os.path.join(decompressed_dir, "SPIHT"), exist_ok=True)

    for i in range(1, max_cases + 1):
        case_name = f"BraTS20_Training_{i:03d}"
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.exists(case_dir):
            continue

        print(f"[Process BraTS] {case_name}")
        imaging_path = find_nifti(os.path.join(case_dir, f"{case_name}_{modality}"))
        if imaging_path is None:
            print(f"[Skip] Missing modality {modality} for {case_name}")
            continue

        volume = nib.load(imaging_path).get_fdata()
        num_slices = volume.shape[2]
        if max_slices:
            num_slices = min(num_slices, max_slices)

        case_comp_dir = os.path.join(compressed_dir, "SPIHT", case_name)
        case_decomp_dir = os.path.join(decompressed_dir, "SPIHT", case_name)
        os.makedirs(case_comp_dir, exist_ok=True)
        os.makedirs(case_decomp_dir, exist_ok=True)

        for s in range(num_slices):
            slice_data = volume[:, :, s]
            if slice_data.max() > slice_data.min():
                normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
            else:
                normalized = np.zeros_like(slice_data)
            normalized = normalized.astype(np.float32)

            output_npz = os.path.join(case_comp_dir, f"slice_{s:03d}.npz")
            encode_full(normalized, output_npz, level, ratio)
            np.savez(os.path.join(case_comp_dir, f"slice_{s:03d}_meta.npz"),
                     norm_min=float(slice_data.min()), norm_max=float(slice_data.max()))

            rec_norm, _ = reconstruct_slice(case_comp_dir, s)
            plt.imsave(os.path.join(case_decomp_dir, f"slice_{s:03d}.png"),
                       rec_norm, cmap="gray", vmin=0, vmax=255)

        print(f"[Done] {case_name}: {num_slices} slices.")




if __name__ == "__main__":
    base_out = "./non_ROI_outputs"
    kits_out = os.path.join(base_out, "outputs_kits")
    brats_out = os.path.join(base_out, "outputs_brats")

    process_kits19_dataset(
        data_dir="./kits19/data",
        compressed_dir=os.path.join(kits_out, "compressed_data"),
        decompressed_dir=os.path.join(kits_out, "decompressed_data"),
        max_cases=2,
        max_slices=3,
        ratio=10
    )

    process_brats_dataset(
        data_dir="./MICCAI_BraTS2020_TrainingData",
        compressed_dir=os.path.join(brats_out, "compressed_data"),
        decompressed_dir=os.path.join(brats_out, "decompressed_data"),
        max_cases=2,
        max_slices=3,
        ratio=10
    )
