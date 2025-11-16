import glymur
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import cv2
import time

"""
Author: Justin Vaughn
Medical Image ROI Compression. Higher quality encoding for regions of interest and more compression for the background.

Based on:
Glymur library docs for JPEG 2000 (https://glymur.readthedocs.io/en/latest/):
Welcome to glymur’s documentation!. Welcome to glymur’s documentation! - glymur 0.14.4 documentation. (n.d.). https://glymur.readthedocs.io/en/latest/
NiBabel for NIfTI format (https://nipy.org/nibabel/):
Nibabel. Neuroimaging in Python - NiBabel 5.4.0.dev1+g3b1c7b37 documentation. (n.d.). https://nipy.org/nibabel/
NumPy array operations and mathematical functions:
https://numpy.org/doc/
NumPy documentation. NumPy. (n.d.). https://numpy.org/doc/


Works with KiTS19 (kidney tumors) and BraTS2021 (brain tumors) datasets.
"""

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
        # Any non-zero value marks abnormal tissue
        return mask_data > 0
    elif isinstance(roi_label, (list, tuple)):
        # Multiple values, check if mask_data is in those values
        return np.isin(mask_data, roi_label)
    else:
        # Single value means exact match
        return mask_data == roi_label

def load_nifti_image_and_mask(image_path, mask_path, slice_idx=None, roi_label=None):
    """
    Load a slice from NIfTI medical image files
    Figured out NIfTI loading from nibabel docs:
    - nib.load(): https://nipy.org/nibabel/reference/nibabel.loadsave.html#load
    - get_fdata(): https://nipy.org/nibabel/reference/nibabel.dataobj_images.html


    Input: 
        image_path - path to .nii.gz image 
        mask_path - path to .nii.gz segmentation
        slice_idx - which slice to extract
        roi_label - label value(s) for ROI
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
    if slice_idx is None:
    # If None: calculate middle slice by floor dividing third dimension size by 2
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
    Min-max normalization formula:
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html    Uses numpy min/max: 
    Input:
        img - numpy array
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
    if img_max == img_min:
        return np.zeros_like(img, dtype=np.uint16)
    else:
        img_norm = (img-img_min) / (img_max-img_min)

    # Scale normalized values
    img_norm=img_norm*max_val


    # return result converted to np.uint16
    return img_norm.astype(np.uint16)

def encode_roi(img, roi_mask, output_path):
    """
    Encodes ROI with high quality
    
    Using glymur for JP2 encoding: https://glymur.readthedocs.io/en/latest/api.html#glymur.Jp2k
    
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
    glymur.Jp2k(output_path, data=img_roi, 
                irreversible=False,
                numres=5)

def encode_bg(img, roi_mask, output_path):
    """
    Higher compression for background, but still lossless encoding.

    Using glymur for JP2 encoding: https://glymur.readthedocs.io/en/latest/api.html#glymur.Jp2k

    Input: 
        image - numpy array
        roi_mask - boolean mask
        output_path - where to save .jp2 file
    """
    # Create copy of input image
    img_bg=img.copy()

    # Zero out roi
    img_bg[roi_mask]=0
    

    # Encode background image to JPEG 2000
    glymur.Jp2k(output_path, data=img_bg, 
                cratios=[30, 20, 10],  
                irreversible=True,
                numres=4)
    
def decode_and_combine(roi_path, bg_path, output_path):
    """
    Decode both files and combine them
    Reading JP2: https://glymur.readthedocs.io/en/latest/how_do_i.html

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
    glymur.Jp2k(output_path, data=combined, 
                irreversible=False,
                numres=5)

def process_3d_volume_slicewise(image_path, mask_path, metrics_dir, output_dir,
roi_label=None, slice_range=None, max_slices=None):
    """
    Process multiple slices from a 3D volume
    
    Figured out NIfTI loading from nibabel docs:
    - nib.load(): https://nipy.org/nibabel/reference/nibabel.loadsave.html#load
    - get_fdata(): https://nipy.org/nibabel/reference/nibabel.dataobj_images.html

    For os:
        https://docs.python.org/3/library/os.html#module-os
    For Path
        https://docs.python.org/3/library/pathlib.html
    For os.path:
        https://docs.python.org/3/library/os.path.html#module-os.path
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
    f = open(metrics_dir, "a", buffering=1)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


    # Extract base filename from image_path:
    base_name=Path(image_path).stem.replace('.nii', '')

    # Load full 3D mask volume
    mask_nifti = nib.load(mask_path)
    mask_data = mask_nifti.get_fdata()
    num_slices = mask_data.shape[2]
    
    processed_count=0


    if slice_range is not None:
        # Use specified slice range
        slice_indices = range(slice_range[0], slice_range[1])
    else:
        # Process ALL slices in the volume
        slice_indices = range(num_slices)

    # Limit number of slices if max_slices is specified
    if max_slices is not None and len(slice_indices) > max_slices:
        # Sample evenly from the slices with ROI
        step = len(slice_indices) // max_slices
        slice_indices = slice_indices[::step][:max_slices]

    # Loop through each slice_idx in slice_indices:
    for slice_idx in slice_indices:
        start_time = time.time()
        if max_slices != None and processed_count >= max_slices:
            break
        
        # Get normalized image and ROI mask for this slice
        img_slice, roi_mask = load_nifti_image_and_mask(
            image_path, mask_path, slice_idx=slice_idx, roi_label=roi_label
        )

        if img_slice.max() == 0:
            continue
                

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
        end_time = time.time()

        f.write(f"Elapsed time for {slice_prefix}: {end_time - start_time}.\n")



    # Print processed_count and base_name 
    # print(f"Processed {processed_count} slices for {base_name}.")
    
    # Return processed_count
    return processed_count

def process_kits19_case(case_dir, output_dir, roi_label=2, max_slices=None, metrics_dir=None):
    """
    Process a single KiTS19 case

    KiTS19 case structure:
    case_dir/
        - imaging.nii.gz (CT scan)
        - segmentation.nii.gz

    Input:
        case_dir - path to individual case folder
        output_dir - output directory
        roi_label - ROI label(s)
        max_slices - maximum number of slices to process per case
    """
    # Extract case name from case_dir
    case_name= os.path.basename(case_dir)
  
    # Construct image file path
    image_path = os.path.join(case_dir, "imaging.nii.gz")
  
    # Construct mask file path
    mask_path = os.path.join(case_dir, "segmentation.nii.gz")
  
    # Check if image_path exists
    if os.path.exists(image_path) == False:
    # If not exists:
    #   Print error message and return
        print("Image file not found.")
        return

    # Check if mask_path exists
    if os.path.exists(mask_path) == False:
    # If not exists:
    #   Print error message and return
        print("Segmentation file not found.")
        return

    print(f"Processing {case_name}")

    # Create case-specific output directory
    case_output_dir=os.path.join(output_dir,case_name)
    # Create directory
    os.makedirs(case_output_dir, exist_ok=True)

    # Call process_3d_volume_slicewise()
    process_3d_volume_slicewise(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=case_output_dir,
        roi_label=roi_label,
        slice_range=None,
        max_slices=max_slices,
        metrics_dir=metrics_dir
    )

    

    # optional, comment out when running on full data sets for experiment
    print(f"Completed processing {case_name}")




def process_brats_case(case_dir, output_dir, roi_label=[1, 2, 4], max_slices=None, metrics_dir = None):
    """
    Process a single BraTS2020 case
    BraTS2020 structure:
    case_dir/BraTS20_Training_001/
        - BraTS20_Training_001_flair.nii
        - BraTS20_Training_001_t1.nii
        - BraTS20_Training_001_t1ce.nii
        - BraTS20_Training_001_t2.nii
        - BraTS20_Training_001_seg.nii (segmentation mask)

    Segmentation labels:
        0 = background (healthy tissue)
        1 = necrotic/non-enhancing tumor core
        2 = peritumoral edema
        4 = enhancing tumor

    Input:
        case_dir - path to individual case folder
        output_dir - output directory
        roi_label - ROI labels (default [1,2,4] for all tumor regions)
        max_slices - maximum number of slices to process per case (None means process all ROI slices)
    """
    modalities = ["flair", "t1", "t1ce", "t2"]
    image_paths = []
    
    # Extract case name from case_dir
    case_name = os.path.basename(case_dir)
    
    # Construct image file paths for all modalities - ONLY ADD IF THEY EXIST
    for mode in modalities:
        img_path = os.path.join(case_dir, f"{case_name}_{mode}.nii.gz")
        if os.path.exists(img_path):
            image_paths.append(img_path)
    
    # Construct mask file path
    mask_path = os.path.join(case_dir, f"{case_name}_seg.nii.gz")
    
    # Check if any image files were found
    if len(image_paths) == 0:
        print(f"Image files not found for {case_name}")
        return
    
    # Check if mask_path exists
    if not os.path.exists(mask_path):
        print(f"Segmentation file not found for {case_name}: {mask_path}")
        return
    
    # optional, comment out when running on full data sets for experiment
    print(f"Processing {case_name} with {len(image_paths)} modalities")
    
    # Create case-specific output directory
    case_output_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_output_dir, exist_ok=True)
    
    # Call process_3d_volume_slicewise() for each modality
    for image_path in image_paths:
        try:
            num_processed = process_3d_volume_slicewise(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=case_output_dir,
                roi_label=roi_label,
                slice_range=None,
                max_slices=max_slices,
                metrics_dir=metrics_dir
            )
            # print(f"  Processed {Path(image_path).stem}: {num_processed} slices")
        except Exception as e:
            print(f"  Error processing {Path(image_path).stem}: {str(e)}")
    
    # optional, comment out when running on full data sets for experiment
    print(f"Completed processing {case_name}")



def save_viewable_outputs(output_prefix, output_dir=None):
    """
    Convert JP2 and NPY to viewable PNG format
    cv2.imwrite docs: https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html


    Input:
        output_prefix - base filename without extension
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base_filename = os.path.basename(output_prefix)
        png_prefix = os.path.join(output_dir, base_filename)
    else:
        png_prefix = output_prefix

    # Construct combined JP2 path
    combined_path=f"{output_prefix}.jp2"

    if os.path.exists(combined_path):
        jp2_obj = glymur.Jp2k(combined_path)
        image_data = jp2_obj[:]
        cv2.imwrite(f"{png_prefix}_combined.png", image_data)
        print(f"Saved {png_prefix}_combined.png")
    else:
        print(f"Not found: {combined_path}")


    # Construct mask NPY path using f"{output_prefix}_mask.npy"
    mask_path=f"{output_prefix}_mask.npy"


    # Check if mask_path exists
    if os.path.exists(mask_path):
        # Load mask array
        bool_mask=np.load(mask_path)
        # Convert boolean mask to 0-255 uint8
        mask_img = (bool_mask * 255).astype(np.uint8)
        # Save as PNG
        cv2.imwrite(f"{png_prefix}_mask.png", mask_img)
        # Print confirmation message
        print(f"Saved: {png_prefix}_mask.png")
    else:
        print(f"Not found: {mask_path}")


    # Construct ROI JP2 path
    roi_path=f"{output_prefix}_roi.jp2"



    # Check if roi_path exists
    if os.path.exists(roi_path):
        # Open JP2 file by creating glymur.Jp2k object
        roi_obj=glymur.Jp2k(roi_path)
        # Decode image using bracket indexing [:]
        roi_data= roi_obj[:]
        # Save as PNG
        cv2.imwrite(f"{png_prefix}_roi.png", roi_data)
        print(f"Saved: {png_prefix}_roi.png")
    else:
        print(f"Not found: {roi_path}")

    # Construct background JP2 path
    bg_path= f"{output_prefix}_bg.jp2"


    # Check if bg_path exists using os.path.exists():
    if os.path.exists(bg_path):
    # If exists:
    #   Open JP2 file by creating glymur.Jp2k object
        bg_obj=glymur.Jp2k(bg_path)
    #   Decode image using bracket indexing [:]
        bg_data= bg_obj[:]
    #   Save as PNG
        cv2.imwrite(f"{png_prefix}_bg.png", bg_data)
        print(f"Saved: {png_prefix}_bg.png")
    else:
        print(f"Not found: {bg_path}")
    

def process_kits19_dataset(data_dir, output_dir, max_cases=None, roi_label=2, max_slices=None,  metrics_dir=None):
    """
    Batch process KiTS19 cases using explicit iteration through case numbers

    Input:
        data_dir - KiTS19 data folder
        output_dir - output directory
        max_cases - limit number of cases
        roi_label - ROI label(s)
        max_slices - slices per case
    """
    # If max_cases is None, set it to 300 (KiTS19 has case_00000 through case_00299)
    if max_cases is None:
        max_cases=300

    # Loop through case numbers from 0 to max_cases (exclusive):
    for i in range(max_cases):
        case_name = f"case_{i:05d}"
        # Construct full case directory path
        case_dir=os.path.join(data_dir, case_name)
        # Check if case_dir exists
        if not os.path.exists(case_dir):
            print("Case doesn't exist")
            continue
        try:
            process_kits19_case(
                case_dir=case_dir,
                output_dir=output_dir,
                roi_label=roi_label,
                max_slices=max_slices,
                metrics_dir=metrics_dir
            )
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")
        




def process_brats_dataset(data_dir, output_dir, max_cases=None, roi_label=[1, 2, 4], max_slices=None, metrics_dir= None):
    """
    Batch process BraTS2020 cases using explicit iteration through case numbers

    Works with official BraTS2020 structure:
    data_dir/BraTS20_Training_001/
    - BraTS20_Training_001_flair.nii
    - BraTS20_Training_001_t1.nii
    - BraTS20_Training_001_t1ce.nii
    - BraTS20_Training_001_t2.nii
    - BraTS20_Training_001_seg.nii

    Input:
    data_dir - BraTS data folder
    output_dir - output directory 
    max_cases - number of cases to process
    roi_label - ROI labels
    max_slices - slices per case
    """
    # If max_cases is None, set it to 369 (BraTS2020 Training Data has 369 training cases)
    if max_cases==None:
        max_cases=369

    # Loop through case numbers from 0 to max_cases (exclusive):
    for i in range(1,max_cases+1):
        case_name = f"BraTS2021_{i:05d}"
    #   Construct full case directory path
        case_dir=os.path.join(data_dir, case_name)
    #   Check if case_dir exists
    #   If not exists:
        if os.path.exists(case_dir) == False:
            #Print warning message that case doesn't exist, continue to next case
            print("Case doesn't exist")
            continue
        try:

            process_brats_case(
                case_dir=case_dir,
                output_dir=output_dir,
                roi_label=roi_label,
                max_slices=max_slices,
                metrics_dir=metrics_dir
            )
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")

if __name__ == "__main__":



    process_brats_dataset(data_dir="/Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/BraTS2021_Training_Data", 
                          output_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats",
                            max_cases=None, roi_label=[1, 2, 4], max_slices=None,                           
                metrics_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats/jp2_metric_brats.txt")

    
    process_kits19_dataset(data_dir="/Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/kits19-master/data", 
                           output_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits",
                           max_cases=100, roi_label=[2], max_slices=None,
                           metrics_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits/jp2_metric_kits.txt")

