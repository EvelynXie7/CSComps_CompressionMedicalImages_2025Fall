"""
spiht_roi_pipeline.py

ROI-based compression pipeline matching JPEG2000 approach
Uses DWT and SPIHT functions for fair comparison
Adpated from the code from Justin

Author: Evelyn Xie
"""

import sys
import pathlib

# Setup paths to find kits19 module
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))           # For local modules (dwt, SPIHT_encoder, etc.)
sys.path.insert(0, str(ROOT.parent))  

import numpy as np
import nibabel as nib
import os

from dwt import *
from SPIHT_encoder import *
from SPIHT_decoder import func_MySPIHT_Dec
from kits19.starter_code.visualize import hu_to_grayscale, DEFAULT_HU_MIN, DEFAULT_HU_MAX

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
    
    # roi_pixels = np.sum(roi_mask)  # Count ROI pixels
    # print(roi_pixels)
    # original_bits = roi_pixels * 32  # Use 32 bits per pixel for safety
    # max_bits = int(original_bits / compression_ratio)
    original_bits = img.size * 8
    max_bits = int(original_bits / compression_ratio)
    #print(f"ROI encoding: {roi_pixels} pixels, max_bits={max_bits}")
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

def _decode_bitstream_to_spatial(bitstream, level, pad_hw):
    """SPIHT decode -> inverse DWT -> unpad to original HxW (float32, 0..255)."""
    m_wave = func_MySPIHT_Dec(bitstream)   # wavelet domain coeffs (H x W)
    rec = decodeDWT(m_wave, level)         # back to spatial
    rec = unpad(rec, tuple(pad_hw))        # remove padding added before DWT
    return rec.astype(np.float32)

def reconstruct_slice(case_output_dir, slice_idx):
    """Load ROI/BG streams + meta, decode them, and merge to one slice."""
    import numpy as np, os

    roi_npz = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
    bg_npz  = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
    meta_npz= os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")

    if not (os.path.exists(roi_npz) and os.path.exists(bg_npz) and os.path.exists(meta_npz)):
        raise FileNotFoundError("Missing ROI/BG/meta for slice", slice_idx)

    roi = np.load(roi_npz)
    bg  = np.load(bg_npz)
    meta= np.load(meta_npz)
    roi_mask = meta["roi_mask"]
    rec_roi = _decode_bitstream_to_spatial(roi["bitstream"], int(roi["level"]), roi["pad_hw"])
    rec_bg  = _decode_bitstream_to_spatial(bg["bitstream"],  int(bg["level"]),  bg["pad_hw"])
    # import matplotlib.pyplot as plt
    # plt.imsave(f"debug_roi_slice_{slice_idx:03d}.png", rec_roi, cmap="gray", vmin=0, vmax=255)
    # plt.imsave(f"debug_bg_slice_{slice_idx:03d}.png", rec_bg, cmap="gray", vmin=0, vmax=255)
    # plt.imsave(f"debug_mask_slice_{slice_idx:03d}.png", roi_mask.astype(float), cmap="gray")
    
    # print(f"ROI stats: min={rec_roi.min():.2f}, max={rec_roi.max():.2f}, mean={rec_roi.mean():.2f}")
    # print(f"BG stats: min={rec_bg.min():.2f}, max={rec_bg.max():.2f}, mean={rec_bg.mean():.2f}")
    
    
    # merge in normalized domain (you encoded normalized 0..255):
    merged_norm = np.where(roi_mask, rec_roi, rec_bg)
    merged_norm = np.clip(merged_norm, 0, 255)
    # Optionally denormalize back to the original slice scale:
    norm_min = float(meta["norm_min"])
    norm_max = float(meta["norm_max"])
    # if norm_max > norm_min:
    #     merged = (merged_norm / 255.0) * (norm_max - norm_min) + norm_min
    # else:
    #     merged = np.zeros_like(merged_norm, dtype=np.float32)
    #return merged_norm.astype(np.float32), merged.astype(np.float32)
    norm_min = float(meta["norm_min"])
    norm_max = float(meta["norm_max"])
    merged_hu = (merged_norm / 255.0) * (norm_max - norm_min) + norm_min

    return merged_norm.astype(np.float32), merged_hu.astype(np.float32)

    

def show_slice(img2d, title="Reconstructed (0..255)"):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img2d, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.show()



def process_kits19_case(case_dir, output_dir, roi_label=2, max_slices=None,
                        roi_ratio=2, bg_ratio=20, level=2):
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
    
    num_slices = imaging_volume.shape[0]
    if max_slices:
        num_slices = min(num_slices, max_slices)
    
    for slice_idx in range(num_slices):
        slice_data = imaging_volume[slice_idx, :, :]
        slice_data = np.clip(slice_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        normalized = ((slice_data - DEFAULT_HU_MIN) / (DEFAULT_HU_MAX - DEFAULT_HU_MIN) * 255).astype(np.float32)
        
        roi_mask_slice = roi_mask_volume[slice_idx, :, :]
        
        # if slice_data.max() > slice_data.min():
        #     normalized = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min()) * 255
        # else:
        #     normalized = np.zeros_like(slice_data)
        
        # normalized = normalized.astype(np.float32)
        
        roi_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_roi.npz")
        bg_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_bg.npz")
        
        encode_roi(normalized, roi_mask_slice, roi_output, level, roi_ratio)
        encode_bg(normalized, roi_mask_slice, bg_output, level, bg_ratio)
        
        meta_output = os.path.join(case_output_dir, f"slice_{slice_idx:03d}_meta.npz")
        np.savez(meta_output,
                roi_mask=roi_mask_slice,
                norm_min=float(DEFAULT_HU_MIN),  # -512
                norm_max=float(DEFAULT_HU_MAX))  # 512


def process_brats_case(case_dir, output_dir, roi_label=[1, 2, 4], max_slices=None,
                       roi_ratio=2, bg_ratio=20, level=2, modality='t1ce'):
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
        slice_data = imaging_volume[slice_idx,:, :]
        roi_mask_slice = roi_mask_volume[slice_idx,:, :]
        
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
    import pathlib
    import re
    import matplotlib
    matplotlib.use("Agg")  # safe for headless runs
    import matplotlib.pyplot as plt
    # Set output directory
    output_directory = "./output"
    
    # For KiTS19
    process_kits19_dataset(
        data_dir="./kits19/data",           
        output_dir=output_directory,
        max_cases=1,
        roi_label=2,
        max_slices=2
    )
    # Find the just-written case folder
    case_dirs = sorted(
        p for p in pathlib.Path(output_directory).glob("case_*") if p.is_dir()
    )
    if not case_dirs:
        print(f"No case folders found in {output_directory}")
        raise SystemExit

    # regex to extract slice index from file name slice_000_roi.npz
    r_slice = re.compile(r"slice_(\d{3})_roi\.npz$")

    for case_dir in case_dirs:
        # discover all available ROI bitstreams in this case
        roi_files = sorted(case_dir.glob("slice_*_roi.npz"))
        if not roi_files:
            print(f"[Skip] No slices found in {case_dir}")
            continue

        print(f"[Reconstruct] {case_dir.name}: {len(roi_files)} slice(s)")
        for roi_file in roi_files:
            m = r_slice.search(roi_file.name)
            if not m:
                continue
            slice_idx = int(m.group(1))

            try:
                rec_norm, rec_orig = reconstruct_slice(str(case_dir), slice_idx)

                # save a PNG per slice (normalized 0..255)
                png_path = case_dir / f"reconstructed_slice_{slice_idx:03d}.png"
                plt.imsave(png_path, rec_norm, cmap="gray", vmin=0, vmax=255)
                print(f"  - saved {png_path.name}")

            except Exception as e:
                print(f"  ! failed slice {slice_idx:03d}: {e}")


    # case_dirs = [d for d in os.listdir(output_directory) if d.startswith("case_")]
    # if not case_dirs:
    #     print("No encoded outputs found in", output_directory)
    # else:
    #     case_dir = os.path.join(output_directory, sorted(case_dirs)[0])
    #     print("Reconstructing from:", case_dir)

    #     # Decode & merge
    #     rec_norm, rec_orig = reconstruct_slice(case_dir, 0)

    #     # View (normalized 0..255)
    #     try:
    #         show_slice(rec_norm, title="Reconstructed slice (normalized 0..255)")
    #     except Exception as e:
    #         # If you’re on a headless session, save instead of show
    #         import matplotlib.pyplot as plt
    #         plt.imsave(os.path.join(case_dir, "reconstructed_slice.png"), rec_norm, cmap="gray", vmin=0, vmax=255)
    #         print("Saved reconstructed_slice.png to", case_dir)