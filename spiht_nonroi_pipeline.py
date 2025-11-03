"""
SPIHT Pipeline for KiTS19 Dataset
Encodes and decodes medical images using SPIHT compression

Author: Rui Shen
"""

import sys
import pathlib
import numpy as np
import nibabel as nib
from pathlib import Path
import time
import json
import matplotlib.pyplot as plt
from PIL import Image

# Setup paths
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))
from kits19.starter_code.visualize import hu_to_grayscale, DEFAULT_HU_MIN, DEFAULT_HU_MAX
from SPIHT_encoder import func_MySPIHT_Enc
from SPIHT_decoder import func_MySPIHT_Dec
from dwt import runDWT, decodeDWT


def load_nifti_volume(file_path):
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img.affine, img.header
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None


def save_nifti_volume(data, affine, header, output_path):
    """Save data as a NIfTI file"""
    try:
        img = nib.Nifti1Image(data, affine, header)
        nib.save(img, output_path)
        print(f"Saved to {output_path}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")


def pad_to_multiple(img, k, mode="edge"):
    """Pad image to be divisible by k"""
    H, W = img.shape[:2]
    pad_h = (k - (H % k)) % k
    pad_w = (k - (W % k)) % k
    if len(img.shape) == 2:
        padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)
    else:
        padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return padded, (pad_h, pad_w)


def unpad(img, pad_hw):
    """Remove padding from image"""
    pad_h, pad_w = pad_hw
    H, W = img.shape[:2]
    if len(img.shape) == 2:
        return img[:H - pad_h, :W - pad_w]
    else:
        return img[:H - pad_h, :W - pad_w, :]


def normalize_slice(slice_data):
    """Normalize a slice to [0, 255] range"""
    min_val = slice_data.min()
    max_val = slice_data.max()
    if max_val - min_val > 0:
        normalized = (slice_data - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(slice_data)
    return normalized.astype(np.float32), min_val, max_val


def denormalize_slice(normalized_data, min_val, max_val):
    """Denormalize from [0, 255] back to original range"""
    if max_val - min_val > 0:
        original = (normalized_data / 255.0) * (max_val - min_val) + min_val
    else:
        original = np.full_like(normalized_data, min_val)
    return original


def save_slice_as_png(slice_data, output_path, title="", cmap='gray'):
    """
    Save a 2D slice as a PNG image
    slice_data : np.ndarray
        2D array representing the slice
    output_path : Path
        Where to save the PNG
    title : str
        Title for the image
    cmap : str
        Colormap to use
    """
    # Normalize to 0-255
    min_val = slice_data.min()
    max_val = slice_data.max()
    if max_val - min_val > 0:
        normalized = ((slice_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(slice_data, dtype=np.uint8)
    
    # Save using PIL for simple grayscale
    img = Image.fromarray(normalized, mode='L')
    img.save(output_path)
    

def encode_slice(slice_data, level=3, max_bits=50000):
    # Normalize
    normalized, min_val, max_val = normalize_slice(slice_data)
    
    # Pad to power of 2
    k = 2 ** level
    padded, pad_info = pad_to_multiple(normalized, k)
    
    # Apply DWT
    dwt_coeffs = runDWT(padded, level)
    
    # SPIHT encode
    bitstream = func_MySPIHT_Enc(dwt_coeffs, max_bits=max_bits, level=level)
    
    return bitstream, pad_info, min_val, max_val, padded.shape


def decode_slice(bitstream, pad_info, min_val, max_val, original_shape, level=3):
    # SPIHT decode
    dwt_coeffs = func_MySPIHT_Dec(bitstream)
    
    # Apply inverse DWT
    reconstructed = decodeDWT(dwt_coeffs, level)
    
    # Ensure correct shape
    if reconstructed.shape != original_shape:
        reconstructed = np.resize(reconstructed, original_shape)
    
    # Remove padding
    unpadded = unpad(reconstructed, pad_info)
    
    # Denormalize
    denormalized = denormalize_slice(unpadded, min_val, max_val)
    
    return denormalized


def process_case(case_path, output_dir, level=3, max_bits=50000, slice_range=None):
    """
    Process a single case (encode and decode)
    
    case_path : Path
        Path to the case directory (e.g., case_00000)
    output_dir : Path
        Directory to save outputs
    level : int
        DWT decomposition level
    max_bits : int
        Maximum bits for encoding
    slice_range : tuple or None
        (start, end) to process only specific slices, or None for all
    """
    print(case_path) 
    case_name = case_path.name
    print(f"\n{'='*60}")
    print(f"Processing {case_name}")
    print(f"{'='*60}")
    
    # Load imaging data
    imaging_path = case_path / "imaging.nii.gz"
    if not imaging_path.exists():
        print(f"Error: {imaging_path} not found")
        return
    
    volume_data, affine, header = load_nifti_volume(imaging_path)
    if volume_data is None:
        return
    
    print(f"Volume shape: {volume_data.shape}")
    print(f"Volume range: [{volume_data.min():.2f}, {volume_data.max():.2f}]")
    
    # Determine slices to process
    num_slices = volume_data.shape[0]
    if slice_range is None:
        start_slice, end_slice = 0, num_slices
    else:
        start_slice, end_slice = slice_range
        start_slice = max(0, start_slice)
        end_slice = min(num_slices, end_slice)
    
    print(f"Processing slices {start_slice} to {end_slice-1}")
    
    # Create output directories
    case_output_dir = output_dir / case_name
    case_output_dir.mkdir(parents=True, exist_ok=True)
    bitstream_dir = case_output_dir / "bitstreams"
    bitstream_dir.mkdir(exist_ok=True)
    png_dir = case_output_dir / "png_slices"
    png_dir.mkdir(exist_ok=True)
    
    # Process each slice
    reconstructed_volume = np.zeros_like(volume_data)
    metadata = {}
    total_bits = 0
    
    start_time = time.time()
    
    for slice_idx in range(start_slice, end_slice):
        print(f"\nSlice {slice_idx}/{num_slices-1}...", end=" ")
        
        # Get slice
        slice_data = volume_data[slice_idx, :, :]
        #slice_data = volume_data[:, :, slice_idx]
        slice_data = np.clip(slice_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)
        slice_data = ((slice_data - DEFAULT_HU_MIN) / (DEFAULT_HU_MAX - DEFAULT_HU_MIN) * 255).astype(np.float32)
        
        # Encode
        bitstream, pad_info, min_val, max_val, padded_shape = encode_slice(
            slice_data, level=level, max_bits=max_bits
        )
        
        # Save bitstream and metadata
        bitstream_path = bitstream_dir / f"slice_{slice_idx:04d}.npy"
        np.save(bitstream_path, bitstream)
        
        metadata[f"slice_{slice_idx:04d}"] = {
            "pad_info": pad_info,
            "min_val": float(min_val),
            "max_val": float(max_val),
            "padded_shape": padded_shape,
            "bitstream_size": len(bitstream),
            "original_shape": slice_data.shape
        }
        
        total_bits += len(bitstream)
        
        # Decode
        reconstructed_slice = decode_slice(
            bitstream, pad_info, min_val, max_val, padded_shape, level=level
        )
        
        # Ensure correct shape
        if reconstructed_slice.shape != slice_data.shape:
            reconstructed_slice = reconstructed_slice[:slice_data.shape[0], :slice_data.shape[1]]
        
        reconstructed_volume[slice_idx, :, :] = reconstructed_slice
        #reconstructed_volume[:, :, slice_idx] = reconstructed_slice
        
        # Save PNG images
        # Save original
        original_png_path = png_dir / f"slice_{slice_idx:04d}_original.png"
        save_slice_as_png(slice_data, original_png_path)
        
        # Save reconstructed
        reconstructed_png_path = png_dir / f"slice_{slice_idx:04d}_reconstructed.png"
        save_slice_as_png(reconstructed_slice, reconstructed_png_path)
    
    
    # Save reconstructed volume
    output_path = case_output_dir / f"{case_name}_reconstructed.nii.gz"
    save_nifti_volume(reconstructed_volume, affine, header, output_path)


def main():
    """Main pipeline function"""
    # Configuration
    DATA_DIR = Path("kits19/data")
    OUTPUT_DIR = Path("outputs_nonROI")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Encoding parameters
    LEVEL = 2  # DWT level
    MAX_BITS = 2000000  # Maximum bits per slice (increased for medical images)
    
    # Cases to process (modify as needed)
    CASES_TO_PROCESS = ["case_00000"]  # Add more cases as needed
    
    # Slice range (None for all slices, or (start, end) for specific range)
    SLICE_RANGE = (0,1)
    
    print("SPIHT Encoding/Decoding Pipeline")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"DWT Level: {LEVEL}")
    print(f"Max bits per slice: {MAX_BITS}")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        print(f"Error: Data directory {DATA_DIR} not found!")
        print("Please ensure kits19/data exists with case folders.")
        return
    
    # Process each case
    for case_name in CASES_TO_PROCESS:
        case_path = DATA_DIR / case_name
        if not case_path.exists():
            print(f"Warning: {case_path} not found, skipping...")
            continue
        
        try:
            process_case(case_path, OUTPUT_DIR, level=LEVEL, max_bits=MAX_BITS, slice_range=SLICE_RANGE)
        except Exception as e:
            print(f"Error processing {case_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Pipeline completed!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()