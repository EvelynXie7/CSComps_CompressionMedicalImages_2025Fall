from metrics import getMSE, getPSNRjp2
import os
from pathlib import Path
import glymur
import numpy as np
import nibabel as nib
import os
from pathlib import Path
import cv2
import time
import json
"""
metrics script for medical image compression
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

def metrics_brats_dataset(data_dir, max_cases, output_dir, roi_label, max_slices, metrics_file="metrics.json"):
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

    # Initialize or load existing metrics
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}
    
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

            metrics_brats_case(
                case_dir=case_dir,
                roi_label=roi_label,
                output_dir=output_dir,
                max_slices=max_slices,
                metrics_file=metrics_file,
                all_metrics=all_metrics
            )
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")

    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

def metrics_brats_case(case_dir, output_dir, roi_label=[1, 2, 4], max_slices=None, metrics_file = None, all_metrics=None):
    modalities = ["flair", "t1", "t1ce", "t2"]
    image_paths = []
    
    # Extract case name from case_dir
    case_name = os.path.basename(case_dir)
    
    # Construct image file paths for all modalities
    for mode in modalities:
        #add .gz for lab
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
            num_processed = metrics_3d_volume_slicewise(
                image_path=image_path,
                mask_path=mask_path,
                output_dir=case_output_dir,
                roi_label=roi_label,
                slice_range=None,
                max_slices=max_slices,
                metrics_file=metrics_file,
                all_metrics=all_metrics
            )

            # print(f"  Processed {Path(image_path).stem}: {num_processed} slices")
        except Exception as e:
            print(f"  Error processing {Path(image_path).stem}: {str(e)}")
    
    print(f"Completed processing {case_name}")

def metrics_3d_volume_slicewise(image_path, mask_path, metrics_file, output_dir,
roi_label=None, slice_range=None, max_slices=None, all_metrics=None):
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
    img_nifti = nib.load(image_path)
    print(img_nifti.get_data_dtype())
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    mask_nifti = nib.load(mask_path)

    # Extract base filename from image_path:
    base_name=Path(image_path).stem.replace('.nii', '')

    # Load full 3D mask volume
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

    # Get case name and modality
    case_name=Path(output_dir).name
    
    if base_name == "imaging":
        modality = "ct"
    else:
        modality = base_name.split('_')[-1]

    # Initialize case and modality in all_metrics if needed
    if case_name not in all_metrics:
        all_metrics[case_name] = {}

    if modality not in all_metrics[case_name]:
        all_metrics[case_name][modality] = []

    # Loop through each slice_idx in slice_indices:
    for slice_idx in slice_indices:
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
        # Create combined_output path
        combined_output = os.path.join(output_dir, f"{slice_prefix}.jp2")

        if not os.path.exists(roi_output) or not os.path.exists(bg_output):
            continue

        # Calculate image composition metrics
        total_pixels = img_slice.size
        has_roi = roi_mask.any()
        roi_pixel_count = np.sum(roi_mask)
        roi_percentage = (roi_pixel_count / total_pixels) * 100
        
        # Calculate black background percentage (pixels with value 0)
        black_pixels = np.sum(img_slice == 0)
        black_percentage = (black_pixels / total_pixels) * 100

        # Load compressed images
        bg_decompressed = glymur.Jp2k(bg_output)[:].astype(np.uint16)

        original_size = img_slice.nbytes  # raw uint16 image bytes
        roi_size = os.path.getsize(roi_output)
        bg_size  = os.path.getsize(bg_output)
        combined_size = os.path.getsize(combined_output)  # for info only

        # Calculate original sizes
        roi_original_size = roi_pixel_count * 2
        bg_pixel_count = total_pixels - roi_pixel_count
        bg_original_size = bg_pixel_count * 2 

        # overall CR
        cr_two_stream = original_size / (roi_size + bg_size)

        # Region-specific compression ratios
        cr_bg_only  = bg_original_size / bg_size if bg_size > 0 else 0
        cr_roi_only = roi_original_size / roi_size if roi_size > 0 and roi_original_size > 0 else 0

        cr_combined= original_size/combined_size


        # Background metrics
        bg_mask = ~roi_mask
        bg_mse = getMSE(img_slice[bg_mask], bg_decompressed[bg_mask])
        bg_psnr = getPSNRjp2(img_slice[bg_mask], bg_decompressed[bg_mask])
        

        # Increment processed_count by 1
        processed_count+=1

        metrics_entry = {
            "slice_idx": int(slice_idx),
            "has_roi": bool(has_roi),
            "roi_percentage": float(roi_percentage),
            "black_percentage": float(black_percentage),
            "bg_mse": float(bg_mse),
            "bg_psnr": float(bg_psnr),
            "bg_cr": float(cr_bg_only),
            "bg_original_bytes": int(bg_original_size),
            "roi_cr": float(cr_roi_only),
            "roi_original_bytes": int(roi_original_size),
            "compression_ratio_two_stream": float(cr_two_stream),
            "bg_only_bytes": int(bg_size),
            "roi_only_bytes": int(roi_size),
            "combined_bytes": int(combined_size),
            "compression_ratio_combined": float(cr_combined)
        }

        all_metrics[case_name][modality].append(metrics_entry)
        
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
    return processed_count

def process_kits19_case(case_dir, output_dir, roi_label=2, max_slices=None, metrics_file = None, all_metrics=None):
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
    metrics_3d_volume_slicewise(
        image_path=image_path,
        mask_path=mask_path,
        output_dir=case_output_dir,
        roi_label=roi_label,
        slice_range=None,
        max_slices=max_slices,
        metrics_file=metrics_file,
        all_metrics=all_metrics
    )

    # optional, comment out when running on full data sets for experiment
    print(f"Completed processing {case_name}")

def metrics_kits19_dataset(data_dir, output_dir, roi_label=2, max_slices=None, metrics_file="metrics.json", max_cases=None):
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

    # Initialize or load existing metrics
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)
    else:
        all_metrics = {}

    # Loop through case numbers from 0 to max_cases:
    for i in range(max_cases):
        case_name = f"case_{i:05d}"
        # Construct full case directory path
        case_dir_path = os.path.join(data_dir, case_name)
        # Check if case_dir exists
        if not os.path.exists(case_dir_path):
            print(f"Case doesn't exist: {case_name}")
            continue
        try:
            process_kits19_case(
                case_dir=case_dir_path,
                output_dir=output_dir,
                roi_label=roi_label,
                max_slices=max_slices,
                metrics_file=metrics_file,
                all_metrics=all_metrics
            )
            
        except Exception as e:
            print(f"Error processing {case_name}: {str(e)}")
    
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

if __name__ == "__main__":
    metrics_brats_dataset(data_dir="/Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/BraTS2021_Training_Data", 
                          output_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats",
                            max_cases=4, roi_label=[1, 2, 4], max_slices=None,                           
                   metrics_file="/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats/metrics_brats.json"

)

    
    metrics_kits19_dataset(data_dir="Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/kits19-master/data", 
                            output_dir="/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits",
                           max_cases=1, roi_label=[2], max_slices=None,
                           metrics_file="/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits/metrics_kits.json"
                    )
    
        # process_brats_dataset(data_dir="/Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/BraTS2021_Training_Data", 
    #                       output_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats",
    #                         max_cases=None, roi_label=[1, 2, 4], max_slices=None,                           
    #             metrics_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_brats/jp2_metric_brats.txt")

    
    # process_kits19_dataset(data_dir="/Users/mic/Desktop/MIC_Comps/CSComps_CompressionMedicalImages_2025Fall/kits19-master/data", 
    #                        output_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits",
    #                        max_cases=1, roi_label=[2], max_slices=None,
    #                        metrics_dir= "/Users/mic/Desktop/MIC_Comps/outputs/outputs_kits/jp2_metric_kits.txt")


# rm /Users/justinvaughn/data/metrics/metrics.json
