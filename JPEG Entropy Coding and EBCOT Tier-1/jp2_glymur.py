import glymur
import numpy as np
import nibabel as nib
import os
from pathlib import Path
"""
Author: Justin Vaughn
Medical Image ROI Compression. Higher quality encoding for regions of interest (tumors, lesions) and
more compression for background.

Based on:
Glymur library docs for JPEG 2000 (https://glymur.readthedocs.io/en/latest/)
NiBabel for NIfTI format (https://nipy.org/nibabel/)
Differential ROI concept from Christopoulos et al. 2000, "The JPEG 2000
still image coding system: an overview"

Works with KiTS19 (kidney tumors) and BraTS2020 (brain tumors) datasets.
"""

def create_roi_mask(mask_data, roi_label):
    """
    Create boolean ROI mask from labeled mask data.
    Single source of truth for ROI extraction logic.
    
    Input:
        mask_data (np.ndarray) - Mask array with integer labels
        roi_label (None, int, or list/tuple) - Which label(s) to treat as ROI
    Output:
        np.ndarray: Boolean array where True = ROI pixels
    """
    if roi_label is None:
        # Any non-zero value marks abnormal tissue
        return mask_data > 0
    elif isinstance(roi_label, (list, tuple)):
        # Multiple values - check if mask_slice is in those values
        return np.isin(mask_data, roi_label)
    else:
        # Single value - exact match
        return mask_data == roi_label

def load_nifti_image_and_mask(image_path, mask_path, slice_idx=None, roi_label=None):
    """
    Load a slice from NIfTI medical image files
    Figured out NIfTI loading from nibabel docs:
    - nib.load(): https://nipy.org/nibabel/reference/nibabel.loadsave.html#load
    - get_fdata(): https://nipy.org/nibabel/reference/nibabel.dataobj_images.html

    The np.isin() trick for multiple labels came from numpy docs:
    https://numpy.org/doc/stable/reference/generated/numpy.isin.html

    Input: 
        image_path - path to .nii.gz image 
        mask_path - path to .nii.gz segmentation
        slice_idx - which slice to extract (None picks middle)
        roi_label - label value(s) for ROI (tumor/lesion)
    Output:
        image slice as uint16, boolean ROI mask
    """
    # Load the NIfTI image file 
    img= nib.load(image_path)

    # Load the NIfTI mask file
    mask = nib.load(mask_path)

    # Get numerical data array from image object
    img_data= img.get_fdata()

    # Get numerical data array from mask object
    mask_data= mask.get_fdata()

    # Check if slice_idx is None
    if slice_idx == None:
    # If None: calculate middle slice by integer dividing third dimension size by 2
        slice_idx=img_data.shape[2]//2
    # If not None: use the provided slice_idx value


    # Extract 2D image slice 
    img_slice = img_data[:,:, slice_idx]
    # Extract 2D mask slice
    mask_slice= mask_data[:,:, slice_idx]

    # Normalize image slice to 16-bit range
    img_slice=normalize_image(img_slice)

    # Create boolean ROI mask based on roi_label:
    roi_mask=create_roi_mask(mask_slice, roi_label)


    # Convert ROI mask to boolean data type
    roi_mask = roi_mask.astype(bool)

    # Return the uint16 image slice and boolean ROI mask as tuple
    return img_slice, roi_mask

def normalize_image(img):
    """
    Min-max normalization to scale image intensity
    Standard min-max formula from 
    https://medium.com/@susanne.schmid/image-normalization-in-medical-imaging-f586c8526bd1
    Uses numpy min/max: https://numpy.org/doc/stable/reference/generated/numpy.ndarray.min.html

    Input:
        img - numpy array
        target_dtype - np.uint8 or np.uint16
    Output:
        normalized image in target range
    """
    # Determine max_val based on target bit depth
    max_val=65535

    # Calculate minimum value
    img_min=img.min()

    # Calculate maximum value
    img_max=img.max()

    # min-max normalization:
    img_norm = (img-img_min) / (img_max-img_min)

    # Scale normalized values
    img_norm*max_val

    # return result converted to np.uint16
    return img_norm.dtarget_type(np.uint16)

def encode_roi(img, roi_mask, output_path):
    """
    High quality encoding for ROI (lossless/near-lossless)
    Using glymur for JP2 encoding: https://glymur.readthedocs.io/en/latest/how_do_i.html
    cratios parameter documented here: https://glymur.readthedocs.io/en/latest/api.html#glymur.Jp2k

    The differential quality idea comes from Christopoulos et al. 2000 paper on JPEG 2000

    Input: 
        image - numpy array
        roi_mask - boolean mask
        output_path - where to save .jp2 file
    """
    # Create copy of input image
    img_roi=img.copy()

    # Zero out background
    img_roi[~roi_mask]=0

    # Encode ROI image to JPEG 2000
    glymur.jp2k(output_path,data=img_roi, 
                cratios=[5, 3, 2, 1], 
                irreversible=False, 
                numres=5)

def encode_bg(img, roi_mask, output_path):
    """
    Higher compression for background (lossy is fine here)
    Same glymur approach but with irreversible=True for lossy compression

    Sources:
    Using glymur for JP2 encoding: https://glymur.readthedocs.io/en/latest/how_do_i.html
    cratios parameter documented here: https://glymur.readthedocs.io/en/latest/api.html#glymur.Jp2k

    The differential quality idea comes from Christopoulos et al. 2000 paper on JPEG 2000

    Input: 
        image - numpy array
        roi_mask - boolean mask
        output_path - output file
    """
    # Create copy of input image
    img_bg=img.copy()

    # Zero out roi
    img_bg[roi_mask]=0
    

    # Encode background image to JPEG 2000
    glymur.jp2k(output_path,data=img_bg, 
                cratios=[50, 30, 20, 10], 
                irreversible=False, 
                numres=4)
    
def decode_and_combine(roi_path, bg_path, output_path):
    """
    Decode both files and combine them
    Reading JP2: https://glymur.readthedocs.io/en/latest/how_do_i.html#how-do-i-read-images
    The [:] indexing decodes the image: https://glymur.readthedocs.io/en/latest/api.html

    Input:
        roi_path: ROI .jp2 file
        bg_path: background .jp2 file
        output_path: combined output
    """
    # Open ROI JPEG 2000 file
    jp2_roi=glymur.Jp2k(roi_path)

    # Open background JPEG 2000 file
    jp2_bg=glymur.Jp2k(bg_path)

    # Decode ROI image 
    roi=jp2_roi[:]

    # Decode background image
    bg=jp2_bg[:]

    # Combine decoded images
    # Works because ROI has zeros where background has data and vice versa
    combined=roi+bg

    # Re-encode combined result to JPEG 2000
    glymur.jp2k(output_path, data=combined, 
                cratios=[50, 30, 20, 10], 
                irreversible=False, 
                numres=4)

def process_3d_volume_slicewise(image_path, mask_path, output_dir,
roi_label=None, slice_range=None, max_slices=None):
    """
    Process multiple slices from a 3D volume
    Using nibabel to load full volumes, then np.any() and np.where() to find 
    slices that contain ROI:
    - https://numpy.org/doc/stable/reference/generated/numpy.any.html
    - https://numpy.org/doc/stable/reference/generated/numpy.where.html

    pathlib.Path for filename extraction: https://docs.python.org/3/library/pathlib.html

    Input:
        image_path - .nii.gz image file
        mask_path - .nii.gz mask file
        output_dir - where to save outputs
        roi_label - which label(s) to use as ROI
        slice_range - specific range or None for auto-detect
        max_slices - limit on number of slices
    Output:
        number of slices processed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load full 3D mask volume
    mask_nifti = nib.load(mask_path)
    num_slices = mask_nifti.shape[2]
    slice_indices = range(num_slices)

    # Extract base filename from image_path:
    base_name=Path(image_path).stem.replace('.nii', '')

    # Initialize processed_count
    processed_count=0

    # Loop through each slice_idx in slice_indices:
    for slice_idx in slice_indices:
        if max_slices != None and processed_count >= max_slices:
            break
        
        # Get normalized image and ROI mask for this slice
        img_slice, roi_mask = load_nifti_image_and_mask(
            image_path, mask_path, slice_idx=slice_idx, roi_label=roi_label
        )
        

        # Construct output filenames:
        
        # Create slice_prefix
        slice_prefix = f"{base_name}_slice{slice_idx:03d}"
        # Create roi_output path
        roi_output = os.path.join(output_dir, f"{slice_prefix}_roi.jp2")
        # Create bg_output path
        bg_output = os.path.join(output_dir, f"{slice_prefix}_bg.jp2")
        # Create mask_output path 
        mask_output = os.path.join(output_dir, f"{slice_prefix}_mask.npy")

        # Create combined_output path
        combined_output = os.path.join(output_dir, f"{slice_prefix}.jp2")

        # Encode ROI
        encode_roi(img_slice, roi_mask, roi_output)
        # Encode background
        encode_bg(img_slice, roi_mask, bg_output)
        # Save mask array
        np.save(mask_output,roi_mask)
        # Combine encoded files
        decode_and_combine(roi_output, bg_output,combined_output)
        # Increment processed_count by 1
        processed_count+=1

    # Print processed_count and base_name 
    print(f"Processed {processed_count} slices for {base_name}.")
    
    # Return processed_count
    return processed_count

