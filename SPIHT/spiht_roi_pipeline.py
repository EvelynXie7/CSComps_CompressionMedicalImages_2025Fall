"""
spiht_roi_pipeline.py

ROI-based compression pipeline matching JPEG2000 approach
Uses DWT and SPIHT functions for fair comparison
Adpated from the code from Justin

Author: Evelyn Xie
"""

import numpy as np
import nibabel as nib
import os

from dwt import *
from SPIHT_encoder import *


def create_roi_mask(mask_data, roi_label):
    """
    Create boolean ROI mask from labeled mask data.
    
    Input:
        mask_data (np.ndarray) - Mask array with integer labels
        roi_label (None, int, or list/tuple) - Which label(s) to treat as ROI
    Output:
        np.ndarray: Boolean array where True = ROI pixels
    """
    if roi_label is None:
        return mask_data > 0
    elif isinstance(roi_label, (list, tuple)):
        return np.isin(mask_data, roi_label)
    else:
        return mask_data == roi_label


def load_nifti_image_and_mask(image_path, mask_path, slice_idx=None, roi_label=None):
    """
    Load a slice from NIfTI medical image files
    
    Input:
        image_path (str) - Path to imaging NIfTI file
        mask_path (str) - Path to segmentation NIfTI file
        slice_idx (int or None) - Which slice to load. If None, load full 3D volume
        roi_label (None, int, or list/tuple) - Which label(s) to treat as ROI
    
    Output:
        image_data (np.ndarray) - Image data (2D if slice_idx given, 3D otherwise)
        roi_mask (np.ndarray) - Boolean ROI mask (same shape as image_data)
    """
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    
    image_volume = image_nii.get_fdata()
    mask_volume = mask_nii.get_fdata()
    
    if slice_idx is not None:
        image_data = image_volume[:, :, slice_idx]
        mask_data = mask_volume[:, :, slice_idx]
    else:
        image_data = image_volume
        mask_data = mask_volume
    
    roi_mask = create_roi_mask(mask_data, roi_label)
    
    return image_data, roi_mask


def encode_roi(img, roi_mask, output_path, level=3, compression_ratio=2):
    """
    Encodes ROI with high quality using SPIHT
    
    Input:
        img - numpy array
        roi_mask - boolean mask
        output_path - where to save compressed data
        level - wavelet decomposition level
        compression_ratio - target compression ratio (e.g., 2 means 2:1)
    """
    img_roi = img.copy()
    img_roi[~roi_mask] = 0
    
    k = 2 ** level
    img_roi, pad_hw = pad_to_multiple(img_roi, k)
    
    img_roi_dwt = runDWT(img_roi, level)
    
    original_bits = img.size * 8
    max_bits = int(original_bits / compression_ratio)
    
    roi_bitstream = func_MySPIHT_Enc(
        img_roi_dwt,
        max_bits=max_bits,
        level=level
    )

        
    
    
    np.savez_compressed(output_path, 
                       bitstream=roi_bitstream,
                       shape=img.shape,
                       level=level,
                       compression_ratio=compression_ratio,
                       bits_used=len(roi_bitstream),
                       pad_hw=pad_hw)


def encode_bg(img, roi_mask, output_path, level=3, compression_ratio=20):
    """
    Higher compression for background using SPIHT
    
    Input:
        img - numpy array
        roi_mask - boolean mask
        output_path - where to save compressed data
        level - wavelet decomposition level
        compression_ratio - target compression ratio (e.g., 20 means 20:1)
    """
    img_bg = img.copy()
    img_bg[roi_mask] = 0
    
    img_bg, pad_hw = pad_to_multiple(img_bg, 2 ** level)

    img_bg_dwt = runDWT(img_bg, level)
    
    original_bits = img.size * 8
    max_bits = int(original_bits / compression_ratio)
    
    
    
    bg_bitstream = func_MySPIHT_Enc(
        img_bg_dwt,
        max_bits=max_bits,
        level=level
    )

    
    np.savez_compressed(output_path,
                       bitstream=bg_bitstream,
                       shape=img.shape,
                       level=level,
                       compression_ratio=compression_ratio,
                       bits_used=len(bg_bitstream),
                       pad_hw=pad_hw)


def process_kits19_case(case_dir, output_dir, roi_label=2, max_slices=None,
                        roi_ratio=2, bg_ratio=20, level=3):
    """
    Process a single KiTS19 case
    
    Input:
        case_dir - path to case directory
        output_dir - output directory
        roi_label - ROI label(s)
        max_slices - max number of slices to process
        roi_ratio - ROI compression ratio (default 2:1)
        bg_ratio - background compression ratio (default 20:1)
        level - wavelet decomposition level
    """
    case_name = os.path.basename(case_dir)
    
    imaging_path = os.path.join(case_dir, "imaging.nii.gz")
    segmentation_path = os.path.join(case_dir, "segmentation.nii.gz")
    
    if not os.path.exists(imaging_path) or not os.path.exists(segmentation_path):
        return
    
    imaging_volume = nib.load(imaging_path).get_fdata()
    segmentation_volume = nib.load(segmentation_path).get_fdata()
    
    roi_mask_volume = create_roi_mask(segmentation_volume, roi_label)
    
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    num_slices = imaging_volume.shape[2]
    if max_slices:
        num_slices = min(num_slices, max_slices)
    
    for slice_idx in range(num_slices):
        slice_data = imaging_volume[:, :, slice_idx]
        roi_mask_slice = roi_mask_volume[:, :, slice_idx]
        
        if slice_data.max() > slice_data.min():
            normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        else:
            normalized = np.zeros_like(slice_data)
        
        normalized = normalized.astype(np.float32)
        
        roi_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
        bg_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
        
        encode_roi(normalized, roi_mask_slice, roi_output, level, roi_ratio)
        encode_bg(normalized, roi_mask_slice, bg_output, level, bg_ratio)
        
        meta_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")
        np.savez(meta_output,
                roi_mask=roi_mask_slice,
                norm_min=float(slice_data.min()),
                norm_max=float(slice_data.max()))


def process_brats_case(case_dir, output_dir, roi_label=[1, 2, 4], max_slices=None,
                       roi_ratio=2, bg_ratio=20, level=3, modality='t1ce'):
    """
    Process a single BraTS case
    
    Input:
        case_dir - path to case directory
        output_dir - output directory
        roi_label - ROI labels
        max_slices - max number of slices
        roi_ratio - ROI compression ratio
        bg_ratio - background compression ratio
        level - wavelet decomposition level
        modality - MRI modality
    """
    case_name = os.path.basename(case_dir)
    
    imaging_path = os.path.join(case_dir, f"{case_name}_{modality}.nii.gz")
    segmentation_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
    
    if not os.path.exists(imaging_path) or not os.path.exists(segmentation_path):
        return
    
    imaging_volume = nib.load(imaging_path).get_fdata()
    segmentation_volume = nib.load(segmentation_path).get_fdata()
    
    roi_mask_volume = create_roi_mask(segmentation_volume, roi_label)
    
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    num_slices = imaging_volume.shape[2]
    if max_slices:
        num_slices = min(num_slices, max_slices)
    
    for slice_idx in range(num_slices):
        slice_data = imaging_volume[:, :, slice_idx]
        roi_mask_slice = roi_mask_volume[:, :, slice_idx]
        
        if slice_data.max() > slice_data.min():
            normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        else:
            normalized = np.zeros_like(slice_data)
        
        normalized = normalized.astype(np.float32)
        
        roi_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
        bg_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
        
        encode_roi(normalized, roi_mask_slice, roi_output, level, roi_ratio)
        encode_bg(normalized, roi_mask_slice, bg_output, level, bg_ratio)
        
        meta_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")
        np.savez(meta_output,
                roi_mask=roi_mask_slice,
                norm_min=float(slice_data.min()),
                norm_max=float(slice_data.max()))


def process_kits19_dataset(data_dir, output_dir, max_cases=None, roi_label=2, 
                           max_slices=None, roi_ratio=2, bg_ratio=20):
    """
    Batch process KiTS19 cases
    
    Input:
        data_dir - KiTS19 data folder
        output_dir - output directory
        max_cases - limit number of cases
        roi_label - ROI label(s)
        max_slices - slices per case
        roi_ratio - ROI compression ratio
        bg_ratio - background compression ratio
    """
    if max_cases is None:
        max_cases = 300
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    for i in range(max_cases):
        case_name = f"case_{i:05d}"
        case_dir = os.path.join(data_dir, case_name)
        if not os.path.exists(case_dir):
            print(f"[Warning] Case not found: {case_dir}")
            continue
        

        try:
            process_kits19_case(
                case_dir=case_dir,
                output_dir=output_dir,
                roi_label=roi_label,
                max_slices=max_slices,
                roi_ratio=roi_ratio,
                bg_ratio=bg_ratio
            )
            
        except Exception as e:
            print(f"Error in {case_name}: {e}")



def process_brats_dataset(data_dir, output_dir, max_cases=None, roi_label=[1, 2, 4], 
                          max_slices=None, roi_ratio=2, bg_ratio=20):
    """
    Batch process BraTS2020 cases
    
    Input:
        data_dir - BraTS data folder
        output_dir - output directory
        max_cases - number of cases to process
        roi_label - ROI labels
        max_slices - slices per case
        roi_ratio - ROI compression ratio
        bg_ratio - background compression ratio
    """
    if max_cases is None:
        max_cases = 369
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(1, max_cases + 1):
        case_name = f"BraTS20_Training_{i:03d}"
        case_dir = os.path.join(data_dir, case_name)
        
        if not os.path.exists(case_dir):
            continue
        
        try:
            process_brats_case(
                case_dir=case_dir,
                output_dir=output_dir,
                roi_label=roi_label,
                max_slices=max_slices,
                roi_ratio=roi_ratio,
                bg_ratio=bg_ratio
            )
        except Exception as e:
            pass
        
if __name__ == "__main__":
    # Set output directory
    output_directory = "./output/test_run"
    
    # For KiTS19 - First 4 cases
    process_kits19_dataset(
        data_dir="./kits19/data",           
        output_dir=output_directory,
        max_cases=4,
        roi_label=2,
        max_slices=1
    )