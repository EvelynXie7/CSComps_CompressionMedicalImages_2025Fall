
"""
Adapted to work with SPIHT pipeline outputs:
- Loads compressed .npz files (roi, bg, meta)
- Reconstructs images using SPIHT decoder
- Calculates MSE, PSNR, compression ratios

Author: Rui Shen, Claude (Revised from Justin's metrics_jp2.py)
"""

import os
import sys
import pathlib
import json
import numpy as np
import nibabel as nib
from pathlib import Path
# Setup paths
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from dwt import decodeDWT
from SPIHT_encoder import unpad
from SPIHT_decoder import func_MySPIHT_Dec
from kits_visualize import DEFAULT_HU_MIN, DEFAULT_HU_MAX


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_roi_mask(mask_data, roi_label):
    """Create ROI mask from segmentation labels"""
    if roi_label is None:
        return mask_data > 0
    elif isinstance(roi_label, (list, tuple)):
        return np.isin(mask_data, roi_label)
    else:
        return mask_data == roi_label


def _decode_bitstream_to_spatial(bitstream, level, pad_hw):
    """Decode SPIHT bitstream back to spatial domain"""
    if len(bitstream) == 0:
        # Empty bitstream (no ROI in slice)
        return None
    m_wave = func_MySPIHT_Dec(bitstream)
    rec = decodeDWT(m_wave, level)
    rec = unpad(rec, tuple(pad_hw))
    return rec.astype(np.float32)


def reconstruct_slice(case_output_dir, slice_idx):
    """
    Reconstruct slice from compressed SPIHT files
    
    Returns:
        merged_norm: Reconstructed image in [0, 255]
        merged_hu: Reconstructed image in HU values
        roi_mask: Boolean mask of ROI
        success: Whether reconstruction succeeded
    """
    roi_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
    bg_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
    meta_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")

    # Check if all files exist
    if not (os.path.exists(roi_npz) and os.path.exists(bg_npz) and os.path.exists(meta_npz)):
        return None, None, None, False

    try:
        # Load compressed data
        roi = np.load(roi_npz)
        bg = np.load(bg_npz)
        meta = np.load(meta_npz)

        roi_mask = meta["roi_mask"]
        
        # Decode ROI and background
        rec_roi = _decode_bitstream_to_spatial(roi["bitstream"], int(roi["level"]), roi["pad_hw"])
        rec_bg = _decode_bitstream_to_spatial(bg["bitstream"], int(bg["level"]), bg["pad_hw"])
        
        # Handle empty ROI case
        if rec_roi is None:
            merged_norm = rec_bg
        else:
            merged_norm = np.where(roi_mask, rec_roi, rec_bg)
        
        merged_norm = np.clip(merged_norm, 0, 255)

        # Denormalize to original HU values
        norm_min = float(meta["norm_min"])
        norm_max = float(meta["norm_max"])
        merged_hu = (merged_norm / 255.0) * (norm_max - norm_min) + norm_min

        return merged_norm.astype(np.float32), merged_hu.astype(np.float32), roi_mask, True
    
    except Exception as e:
        print(f"  Error reconstructing slice {slice_idx}: {str(e)}")
        return None, None, None, False


def getMSE(original, reconstructed):
    """Calculate Mean Squared Error"""
    if original.size == 0:
        return 0.0
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    return float(mse)


def getPSNR(original, reconstructed, max_val=255.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = getMSE(original, reconstructed)
    if mse == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return float(psnr)


# ============================================================================
# KITS19 METRICS
# ============================================================================

def metrics_kits19_slice(original_slice, compressed_dir, slice_idx, roi_mask):
    """
    Calculate metrics for a single KiTS19 slice
    
    Args:
        original_slice: Original normalized image [0, 255]
        compressed_dir: Directory containing compressed .npz files
        slice_idx: Slice index
        roi_mask: Boolean ROI mask
    
    Returns:
        dict: Metrics for this slice
    """
    # Reconstruct from compressed files
    rec_norm, rec_hu, _, success = reconstruct_slice(compressed_dir, slice_idx)
    
    if not success:
        return None
    
    # Load file sizes
    roi_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_roi.npz")
    bg_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_bg.npz")
    meta_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_meta.npz")
    
    roi_size = os.path.getsize(roi_npz)
    bg_size = os.path.getsize(bg_npz)
    meta_size = os.path.getsize(meta_npz)
    combined_size = roi_size + bg_size #+ meta_size
    
    # Calculate image composition metrics
    total_pixels = original_slice.size
    roi_pixel_count = np.sum(roi_mask)
    roi_percentage = (roi_pixel_count / total_pixels) * 100
    
    # Original sizes (float32 = 4 bytes per pixel)
    original_size = total_pixels * 4  # float32
    roi_original_size = roi_pixel_count * 4
    bg_original_size = (total_pixels - roi_pixel_count) * 4


    
    # Compression ratios
    cr_combined = original_size / combined_size
    cr_roi = roi_original_size / roi_size if roi_size > 0 and roi_original_size > 0 else 0
    cr_bg = bg_original_size / bg_size if bg_size > 0 and bg_original_size > 0 else 0
    
    # Overall metrics (normalized [0, 255] scale)
    overall_mse = getMSE(original_slice, rec_norm)
    overall_psnr = getPSNR(original_slice, rec_norm, max_val=255.0)
    
    # ROI metrics
    if roi_pixel_count > 0:
        roi_mse = getMSE(original_slice[roi_mask], rec_norm[roi_mask])
        roi_psnr = getPSNR(original_slice[roi_mask], rec_norm[roi_mask], max_val=255.0)
    else:
        roi_mse = 0.0
        roi_psnr = float('inf')
    
    # Background metrics
    bg_mask = ~roi_mask
    bg_pixel_count = np.sum(bg_mask)
    if bg_pixel_count > 0:
        bg_mse = getMSE(original_slice[bg_mask], rec_norm[bg_mask])
        bg_psnr = getPSNR(original_slice[bg_mask], rec_norm[bg_mask], max_val=255.0)
    else:
        bg_mse = 0.0
        bg_psnr = float('inf')
    
    metrics = {
        "slice_idx": int(slice_idx),
        "has_roi": bool(roi_pixel_count > 0),
        "roi_percentage": float(roi_percentage),
        
        # Overall metrics
        "overall_mse": float(overall_mse),
        "overall_psnr": float(overall_psnr),
        
        # ROI metrics
        "roi_mse": float(roi_mse),
        "roi_psnr": float(roi_psnr),
        "roi_pixel_count": int(roi_pixel_count),
        
        # Background metrics
        "bg_mse": float(bg_mse),
        "bg_psnr": float(bg_psnr),
        "bg_pixel_count": int(bg_pixel_count),
        
        # Compression ratios
        "compression_ratio_combined": float(cr_combined),
        "roi_compression_ratio": float(cr_roi),
        "bg_compression_ratio": float(cr_bg),
        
        # File sizes (bytes)
        "original_size_bytes": int(original_size),
        "compressed_size_bytes": int(combined_size),
        "roi_compressed_bytes": int(roi_size),
        "bg_compressed_bytes": int(bg_size),
        "meta_bytes": int(meta_size),
        "roi_original_bytes": int(roi_original_size),
        "bg_original_bytes": int(bg_original_size),
    }
    
    return metrics


def metrics_kits19_case(case_dir, compressed_dir, roi_label=2):
    """
    Calculate metrics for all processed slices in a KiTS19 case
    
    Args:
        case_dir: Path to original case data (imaging.nii.gz, segmentation.nii.gz)
        compressed_dir: Path to compressed SPIHT outputs
        roi_label: ROI label value
    
    Returns:
        list: Metrics for each slice
    """
    case_name = os.path.basename(case_dir)
    
    # Load original data
    imaging_path = os.path.join(case_dir, "imaging.nii.gz")
    segmentation_path = os.path.join(case_dir, "segmentation.nii.gz")
    
    if not os.path.exists(imaging_path) or not os.path.exists(segmentation_path):
        print(f"[Skip] Missing files for {case_name}")
        return []
    
    # img_vol = nib.load(imaging_path).get_fdata()
    # seg_vol = nib.load(segmentation_path).get_fdata()
    # mask_vol = create_roi_mask(seg_vol, roi_label)
    
    img_vol = nib.load(imaging_path).get_fdata()
    seg_vol = nib.load(segmentation_path).get_fdata()
    
    # 🔍 DEBUG: Check what's in the segmentation
    print(f"\n=== DEBUG {case_name} ===")
    print(f"Segmentation volume shape: {seg_vol.shape}")
    print(f"Unique labels in volume: {np.unique(seg_vol)}")
    
    # Check specific slices
    for test_slice in [215, 216, 217, 218, 219]:
        if test_slice < seg_vol.shape[0]:
            labels_in_slice = np.unique(seg_vol[test_slice, :, :])
            tumor_pixels = np.sum(seg_vol[test_slice, :, :] == roi_label)
            print(f"  Slice {test_slice}: labels={labels_in_slice}, tumor_pixels={tumor_pixels}")
    
    mask_vol = create_roi_mask(seg_vol, roi_label)
    
    # 🔍 DEBUG: Check mask
    print(f"\nMask volume shape: {mask_vol.shape}")
    for test_slice in [215, 216, 217, 218, 219]:
        if test_slice < mask_vol.shape[0]:
            roi_count = np.sum(mask_vol[test_slice, :, :])
            print(f"  Slice {test_slice} mask: {roi_count} ROI pixels")
    print("===================\n")
    

    # Find which slices were actually compressed
    compressed_slices = []
    for f in os.listdir(compressed_dir):
        if f.endswith("_roi.npz"):
            slice_idx = int(f.split("_")[1])
            compressed_slices.append(slice_idx)
    
    compressed_slices.sort()
    
    if not compressed_slices:
        print(f"[Skip] No compressed slices found for {case_name}")
        return []
    
    print(f"[Process] {case_name}: {len(compressed_slices)} slices")
    
    case_metrics = []
    
    for slice_idx in compressed_slices:
        # Get original slice
        img = img_vol[slice_idx, :, :]
        mask = mask_vol[slice_idx, :, :]
        
        # Normalize same way as pipeline
        img = np.clip(img, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        norm = ((img - DEFAULT_HU_MIN) / (DEFAULT_HU_MAX - DEFAULT_HU_MIN) * 255).astype(np.float32)
        
        # Calculate metrics
        slice_metrics = metrics_kits19_slice(norm, compressed_dir, slice_idx, mask)
        
        if slice_metrics is not None:
            case_metrics.append(slice_metrics)
    
    return case_metrics


def metrics_kits19_dataset(data_dir, compressed_base_dir, output_file, 
                          roi_label=2, max_cases=None):
    """
    Calculate metrics for entire KiTS19 dataset
    
    Args:
        data_dir: Original KiTS19 data directory
        compressed_base_dir: Base directory containing compressed_data/SPIHT/
        output_file: JSON file to save metrics
        roi_label: ROI label value
        max_cases: Maximum number of cases to process (None for all)
    """
    if max_cases is None:
        max_cases = 300
    
    # Initialize or load existing metrics
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    compressed_spiht_dir = os.path.join(compressed_base_dir,"outputs_kits", "compressed_data","SPIHT")
    print (compressed_spiht_dir)
    for i in range(0,max_cases+1):
        case_name = f"case_{i:05d}"
        case_dir = os.path.join(data_dir, case_name)
        print(case_dir)
        compressed_case_dir = os.path.join(compressed_spiht_dir, case_name)
        print(compressed_case_dir)
        if not os.path.exists(case_dir):
            continue
        
        if not os.path.exists(compressed_case_dir):
            print(f"[Skip] No compressed data for {case_name}")
            continue
        
        try:
            case_metrics = metrics_kits19_case(case_dir, compressed_case_dir, roi_label)
            
            if case_metrics:
                all_metrics[case_name] = {
                    "slices": case_metrics,
                    "num_slices": len(case_metrics)
                }
                
                # Calculate case-level averages
                all_metrics[case_name]["averages"] = calculate_averages(case_metrics)
                
                # Save after each case
                with open(output_file, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
                
                print(f"  Processed {len(case_metrics)} slices")
        
        except Exception as e:
            print(f"[Error] {case_name}: {str(e)}")
    
    print(f"\nMetrics saved to: {output_file}")


# ============================================================================
# BRATS METRICS
# ============================================================================

def metrics_brats_slice(original_slice, compressed_dir, slice_idx, roi_mask):
    """
    Calculate metrics for a single BraTS slice
    
    Similar to KiTS19 but handles BraTS normalization
    """
    # Reconstruct from compressed files
    rec_norm, _, _, success = reconstruct_slice(compressed_dir, slice_idx)
    
    if not success:
        return None
    
    # Load file sizes (same as KiTS19)
    roi_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_roi.npz")
    bg_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_bg.npz")
    meta_npz = os.path.join(compressed_dir, f"slice_{slice_idx:03d}_meta.npz")
    
    roi_size = os.path.getsize(roi_npz)
    bg_size = os.path.getsize(bg_npz)
    meta_size = os.path.getsize(meta_npz)
    combined_size = roi_size + bg_size + meta_size
    
    # Calculate metrics (same structure as KiTS19)
    total_pixels = original_slice.size
    roi_pixel_count = np.sum(roi_mask)
    roi_percentage = (roi_pixel_count / total_pixels) * 100
    
    # Original sizes
    original_size = total_pixels * 4
    roi_original_size = roi_pixel_count * 4
    bg_original_size = (total_pixels - roi_pixel_count) * 4
    
    # Compression ratios
    cr_combined = original_size / combined_size
    cr_roi = roi_original_size / roi_size if roi_size > 0 and roi_original_size > 0 else 0
    cr_bg = bg_original_size / bg_size if bg_size > 0 and bg_original_size > 0 else 0
    

    # Quality metrics
    overall_mse = getMSE(original_slice, rec_norm)
    overall_psnr = getPSNR(original_slice, rec_norm, max_val=255.0)
    
    if roi_pixel_count > 0:
        roi_mse = getMSE(original_slice[roi_mask], rec_norm[roi_mask])
        roi_psnr = getPSNR(original_slice[roi_mask], rec_norm[roi_mask], max_val=255.0)
    else:
        roi_mse = 0.0
        roi_psnr = float('inf')
    
    bg_mask = ~roi_mask
    bg_pixel_count = np.sum(bg_mask)
    if bg_pixel_count > 0:
        bg_mse = getMSE(original_slice[bg_mask], rec_norm[bg_mask])
        bg_psnr = getPSNR(original_slice[bg_mask], rec_norm[bg_mask], max_val=255.0)
    else:
        bg_mse = 0.0
        bg_psnr = float('inf')
    
    metrics = {
        "slice_idx": int(slice_idx),
        "has_roi": bool(roi_pixel_count > 0),
        "roi_percentage": float(roi_percentage),
        "overall_mse": float(overall_mse),
        "overall_psnr": float(overall_psnr),
        "roi_mse": float(roi_mse),
        "roi_psnr": float(roi_psnr),
        "roi_pixel_count": int(roi_pixel_count),
        "bg_mse": float(bg_mse),
        "bg_psnr": float(bg_psnr),
        "bg_pixel_count": int(bg_pixel_count),
        "compression_ratio_combined": float(cr_combined),
        "roi_compression_ratio": float(cr_roi),
        "bg_compression_ratio": float(cr_bg),
        "original_size_bytes": int(original_size),
        "compressed_size_bytes": int(combined_size),
        "roi_compressed_bytes": int(roi_size),
        "bg_compressed_bytes": int(bg_size),
        "meta_bytes": int(meta_size),
        "roi_original_bytes": int(roi_original_size),
        "bg_original_bytes": int(bg_original_size),
    }
    
    return metrics


def metrics_brats_case(case_dir, compressed_dir, roi_label=[1, 2, 4], modality="t1ce"):
    """
    Calculate metrics for a BraTS case (single modality)
    
    Args:
        case_dir: Original case directory
        compressed_dir: Compressed SPIHT output directory for this case
        roi_label: ROI labels
        modality: Which modality to process
    """
    case_name = os.path.basename(case_dir)
    
    # Load original data
    img_path = os.path.join(case_dir, f"{case_name}_{modality}.nii.gz")
    seg_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
    
    if not os.path.exists(img_path):
        img_path = img_path.replace(".nii.gz", ".nii")
    if not os.path.exists(seg_path):
        seg_path = seg_path.replace(".nii.gz", ".nii")
    
    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        print(f"[Skip] Missing files for {case_name}")
        return []
    
    img_vol = nib.load(img_path).get_fdata()
    seg_vol = nib.load(seg_path).get_fdata()
    mask_vol = create_roi_mask(seg_vol, roi_label)
    
    # Find compressed slices
    compressed_slices = []
    for f in os.listdir(compressed_dir):
        if f.endswith("_roi.npz"):
            slice_idx = int(f.split("_")[1])
            compressed_slices.append(slice_idx)
    
    compressed_slices.sort()
    
    if not compressed_slices:
        return []
    
    case_metrics = []
    
    for slice_idx in compressed_slices:
        # Get original slice (BraTS: slices along axis 2)
        img = img_vol[:, :, slice_idx]
        mask = mask_vol[:, :, slice_idx]
        
        # Normalize same way as pipeline
        if img.max() > img.min():
            norm = (img - img.min()) / (img.max() - img.min()) * 255
        else:
            norm = np.zeros_like(img)
        norm = norm.astype(np.float32)
        
        # Calculate metrics
        slice_metrics = metrics_brats_slice(norm, compressed_dir, slice_idx, mask)
        
        if slice_metrics is not None:
            case_metrics.append(slice_metrics)
    
    return case_metrics


def metrics_brats_dataset(data_dir, compressed_base_dir, output_file,
                         roi_label=[1, 2, 4], max_cases=None, modality="t1ce"):
    """
    Calculate metrics for BraTS dataset
    """
    if max_cases is None:
        max_cases = 500
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
    compressed_spiht_dir = os.path.join(compressed_base_dir,"outputs_brats", "compressed_data","SPIHT")
    
    for i in range(1, max_cases + 1):
        case_name = f"BraTS2021_{i:05d}"
        case_dir = os.path.join(data_dir, case_name)
        compressed_case_dir = os.path.join(compressed_spiht_dir, case_name)
        
        if not os.path.exists(case_dir):
            continue
        
        if not os.path.exists(compressed_case_dir):
            continue
        
        try:
            case_metrics = metrics_brats_case(case_dir, compressed_case_dir, 
                                            roi_label, modality)
            
            if case_metrics:
                all_metrics[case_name] = {
                    "modality": modality,
                    "slices": case_metrics,
                    "num_slices": len(case_metrics),
                    "averages": calculate_averages(case_metrics)
                }
                
                with open(output_file, 'w') as f:
                    json.dump(all_metrics, f, indent=2)
                
                print(f"[Process] {case_name}: {len(case_metrics)} slices")
        
        except Exception as e:
            print(f"[Error] {case_name}: {str(e)}")
    
    print(f"\nMetrics saved to: {output_file}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_averages(metrics_list):
    """Calculate average metrics across all slices"""
    if not metrics_list:
        return {}
    
    numeric_keys = ['overall_mse', 'overall_psnr', 'roi_mse', 'roi_psnr',
                   'bg_mse', 'bg_psnr', 'compression_ratio_combined',
                   'roi_compression_ratio', 'bg_compression_ratio',
                   'roi_percentage']
    
    averages = {}
    for key in numeric_keys:
        values = [m[key] for m in metrics_list if key in m and m[key] != float('inf')]
        if values:
            averages[f"avg_{key}"] = float(np.mean(values))
    
    # Total sizes
    averages["total_original_bytes"] = sum(m.get("original_size_bytes", 0) for m in metrics_list)
    averages["total_compressed_bytes"] = sum(m.get("compressed_size_bytes", 0) for m in metrics_list)
    
    if averages["total_compressed_bytes"] > 0:
        averages["overall_compression_ratio"] = (
            averages["total_original_bytes"] / averages["total_compressed_bytes"]
        )
    
    return averages


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # metrics_brats_dataset(
    #     data_dir="./BraTS2021_Training_Data",
    #     compressed_base_dir="./outputs_SPIHT",
    #     output_file="./outputs_SPIHT/outputs_brats/spiht_metrics_brats.json",
    #     roi_label=[1, 2, 4],
    #     max_cases=100,
    #     modality="t1ce"
    # )
    metrics_kits19_dataset(
        data_dir="./kits19-master/data",
        compressed_base_dir="./outputs_SPIHT",
        output_file="./outputs_SPIHT/outputs_kits/metrics/spiht_metrics_kits.json",
        roi_label=2,
        max_cases=100
    )
    
