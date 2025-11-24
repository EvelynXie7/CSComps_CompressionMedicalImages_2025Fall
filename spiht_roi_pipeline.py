"""
spiht_roi_pipeline.py

ROI-based compression pipeline matching JPEG2000's ROI approach 
Adpated from the code from Justin Vaughn

Author: Evelyn Xie

"""
import os
import sys
import re
import pathlib
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup paths
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from transforms import *
from SPIHT_encoder import *
from SPIHT_decoder import func_MySPIHT_Dec
from utils import DEFAULT_HU_MIN, DEFAULT_HU_MAX


# Helper Utilities 

def find_nifti(base_path):
    if os.path.exists(base_path + ".nii.gz"):
        return base_path + ".nii.gz"
    elif os.path.exists(base_path + ".nii"):
        return base_path + ".nii"
    return None


def create_roi_mask(mask_data, roi_label):
    if roi_label is None:
        return mask_data > 0
    elif isinstance(roi_label, (list, tuple)):
        return np.isin(mask_data, roi_label)
    else:
        return mask_data == roi_label


def _decode_bitstream_to_spatial(bitstream, level, pad_hw):
    m_wave = func_MySPIHT_Dec(bitstream)
    rec = decodeDWT(m_wave, level)
    rec = unpad(rec, tuple(pad_hw))
    return rec.astype(np.float32)


# Encoding
def encode_roi(img, roi_mask, output_path, level=3, compression_ratio=2):
    img_roi = img.copy()
    img_roi[~roi_mask] = 0
    k = 2 ** level
    img_roi, pad_hw = pad_to_multiple(img_roi, k)
    img_dwt = runDWT(img_roi, level)
    original_bits = img.size * 32
    max_bits = int(original_bits / compression_ratio)
    roi_bitstream = func_MySPIHT_Enc(img_dwt, max_bits=max_bits, level=level)
    np.savez_compressed(output_path, bitstream=roi_bitstream, shape=img.shape,
                        level=level, compression_ratio=compression_ratio,
                        bits_used=len(roi_bitstream), pad_hw=pad_hw)


def encode_bg(img, roi_mask, output_path, level=3, compression_ratio=20):
    img_bg = img.copy()
    img_bg[roi_mask] = 0
    k = 2 ** level
    img_bg, pad_hw = pad_to_multiple(img_bg, k)
    img_dwt = runDWT(img_bg, level)
    original_bits = img.size * 32
    max_bits = int(original_bits / compression_ratio)
    bg_bitstream = func_MySPIHT_Enc(img_dwt, max_bits=max_bits, level=level)
    np.savez_compressed(output_path, bitstream=bg_bitstream, shape=img.shape,
                        level=level, compression_ratio=compression_ratio,
                        bits_used=len(bg_bitstream), pad_hw=pad_hw)


def reconstruct_slice(case_output_dir, slice_idx):
    roi_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
    bg_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
    meta_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")

    if not (os.path.exists(roi_npz) and os.path.exists(bg_npz) and os.path.exists(meta_npz)):
        raise FileNotFoundError(f"Missing ROI/BG/meta for slice {slice_idx}")

    roi = np.load(roi_npz)
    bg = np.load(bg_npz)
    meta = np.load(meta_npz)

    roi_mask = meta["roi_mask"]
    rec_roi = _decode_bitstream_to_spatial(roi["bitstream"], int(roi["level"]), roi["pad_hw"])
    rec_bg = _decode_bitstream_to_spatial(bg["bitstream"], int(bg["level"]), bg["pad_hw"])

    merged_norm = np.where(roi_mask, rec_roi, rec_bg)
    merged_norm = np.clip(merged_norm, 0, 255)

    norm_min = float(meta["norm_min"])
    norm_max = float(meta["norm_max"])
    merged_hu = (merged_norm / 255.0) * (norm_max - norm_min) + norm_min

    return merged_norm.astype(np.float32), merged_hu.astype(np.float32)


# Dataset Pipeline


def process_kits19_dataset(data_dir, compressed_dir, decompressed_dir,
                           max_cases=2, roi_label=2, max_slices=5,
                           roi_ratio=2, bg_ratio=10, level=2):
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
        segmentation_path = os.path.join(case_dir, "segmentation.nii.gz")
        if not os.path.exists(imaging_path) or not os.path.exists(segmentation_path):
            print(f"[Skip] Missing {case_name}")
            continue

        img_vol = nib.load(imaging_path).get_fdata()
        seg_vol = nib.load(segmentation_path).get_fdata()
        mask_vol = create_roi_mask(seg_vol, roi_label)

        num_slices = img_vol.shape[0]
        if max_slices:
            num_slices = min(num_slices, max_slices)

        case_comp_dir = os.path.join(compressed_dir, "SPIHT", case_name)
        case_decomp_dir = os.path.join(decompressed_dir, "SPIHT", case_name)
        os.makedirs(case_comp_dir, exist_ok=True)
        os.makedirs(case_decomp_dir, exist_ok=True)

        for s in range(num_slices):
            img = img_vol[s, :, :]
            mask = mask_vol[s, :, :]
            img = np.clip(img, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
            norm = ((img - DEFAULT_HU_MIN) / (DEFAULT_HU_MAX - DEFAULT_HU_MIN) * 255).astype(np.float32)

            encode_roi(norm, mask, os.path.join(case_comp_dir, f"slice_{s:03d}_roi.npz"), level, roi_ratio)
            encode_bg(norm, mask, os.path.join(case_comp_dir, f"slice_{s:03d}_bg.npz"), level, bg_ratio)
            np.savez(os.path.join(case_comp_dir, f"slice_{s:03d}_meta.npz"),
                     roi_mask=mask, norm_min=float(DEFAULT_HU_MIN), norm_max=float(DEFAULT_HU_MAX))

            rec_norm, _ = reconstruct_slice(case_comp_dir, s)
            plt.imsave(os.path.join(case_decomp_dir, f"slice_{s:03d}.png"), rec_norm, cmap="gray", vmin=0, vmax=255)

        print(f"[Done] {case_name}: {num_slices} slices.")


def process_brats_dataset(data_dir, compressed_dir, decompressed_dir,
                          max_cases=2, roi_label=[1, 2, 4], max_slices=5,
                          roi_ratio=2, bg_ratio=10, level=2, modality="t1ce"):
    """Dataset-level processing for BraTS2021."""
    os.makedirs(os.path.join(compressed_dir, "SPIHT"), exist_ok=True)
    os.makedirs(os.path.join(decompressed_dir, "SPIHT"), exist_ok=True)

    for i in range(1, max_cases + 1):
        case_name = f"BraTS2021_{i:05d}"
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.exists(case_dir):
            continue

        print(f"[Process BraTS] {case_name}")
        img_path = find_nifti(os.path.join(case_dir, f"{case_name}_{modality}"))
        seg_path = find_nifti(os.path.join(case_dir, f"{case_name}_seg"))
        if img_path is None or seg_path is None:
            print(f"[Skip] Missing {case_name}")
            continue

        img_vol = nib.load(img_path).get_fdata()
        seg_vol = nib.load(seg_path).get_fdata()
        mask_vol = create_roi_mask(seg_vol, roi_label)
        num_slices = img_vol.shape[2]
        if max_slices:
            num_slices = min(num_slices, max_slices)

        case_comp_dir = os.path.join(compressed_dir, "SPIHT", case_name)
        case_decomp_dir = os.path.join(decompressed_dir, "SPIHT", case_name)
        os.makedirs(case_comp_dir, exist_ok=True)
        os.makedirs(case_decomp_dir, exist_ok=True)

        for s in range(num_slices):
            img = img_vol[:, :, s]
            mask = mask_vol[:, :, s]
            if img.max() > img.min():
                norm = (img - img.min()) / (img.max() - img.min()) * 255
            else:
                norm = np.zeros_like(img)
            norm = norm.astype(np.float32)

            encode_roi(norm, mask, os.path.join(case_comp_dir, f"slice_{s:03d}_roi.npz"), level, roi_ratio)
            encode_bg(norm, mask, os.path.join(case_comp_dir, f"slice_{s:03d}_bg.npz"), level, bg_ratio)
            np.savez(os.path.join(case_comp_dir, f"slice_{s:03d}_meta.npz"),
                     roi_mask=mask, norm_min=float(img.min()), norm_max=float(img.max()))

            rec_norm, _ = reconstruct_slice(case_comp_dir, s)
            plt.imsave(os.path.join(case_decomp_dir, f"slice_{s:03d}.png"), rec_norm, cmap="gray", vmin=0, vmax=255)

        print(f"[Done] {case_name}: {num_slices} slices.")



if __name__ == "__main__":
    base_out = "./outputs_SPIHT_new"
    kits_out = os.path.join(base_out, "outputs_kits")
    brats_out = os.path.join(base_out, "outputs_brats")
    process_brats_dataset(
        data_dir="./BraTS2021_Training_Data",
        compressed_dir=os.path.join(brats_out, "compressed_data"),
        decompressed_dir=os.path.join(brats_out, "decompressed_data"),
        max_cases=100,
        roi_label=[1, 2, 4],
        max_slices=300
    )
    process_kits19_dataset(
        data_dir="./kits19-master/data",
        compressed_dir=os.path.join(kits_out, "compressed_data"),
        decompressed_dir=os.path.join(kits_out, "decompressed_data"),
        max_cases=100,
        roi_label=2,
        max_slices=300
    )

