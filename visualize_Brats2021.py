"""
BraTS 2021 Dataset Image Slice Extraction and Visualization
This script demonstrates how to load and visualize slice images from the BraTS 2021 dataset.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_brats_case(case_path):
    """
    Load all modalities for a single BraTS case.
    
    Args:
        case_path: Path to the case folder (e.g., 'BraTS2021_00000')
    
    Returns:
        Dictionary containing numpy arrays for each modality
    """
    modalities = {}
    
    # BraTS 2021 has 4 modalities + segmentation mask
    files = {
        't1': '_t1.nii.gz',
        't1ce': '_t1ce.nii.gz',  # T1 with contrast enhancement
        't2': '_t2.nii.gz',
        'flair': '_flair.nii.gz',
        'seg': '_seg.nii.gz'  # segmentation mask (if available)
    }
    
    case_name = os.path.basename(case_path)
    
    for modality, suffix in files.items():
        file_path = os.path.join(case_path, case_name + suffix)
        
        if os.path.exists(file_path):
            # Load the NIfTI file
            nii_img = nib.load(file_path)
            # Get the image data as numpy array
            modalities[modality] = nii_img.get_fdata()
            print(f"Loaded {modality}: shape = {modalities[modality].shape}")
        else:
            print(f"Warning: {file_path} not found")
    
    return modalities


def extract_slice(volume, slice_idx, axis=2):
    """
    Extract a 2D slice from a 3D volume.
    
    Args:
        volume: 3D numpy array (typically 240x240x155 for BraTS)
        slice_idx: Index of the slice to extract
        axis: Which axis to slice along (0=sagittal, 1=coronal, 2=axial)
    
    Returns:
        2D numpy array representing the slice
    """
    if axis == 0:  # Sagittal
        return volume[slice_idx, :, :]
    elif axis == 1:  # Coronal
        return volume[:, slice_idx, :]
    else:  # Axial (default)
        return volume[:, :, slice_idx]


def visualize_all_modalities(modalities, slice_idx=77, axis=2, save_path=None):
    """
    Visualize all modalities for a single slice.
    
    Args:
        modalities: Dictionary of modality data
        slice_idx: Which slice to display (default: 77, middle of brain)
        axis: Which axis to slice along (default: 2 for axial view)
        save_path: Optional path to save the figure
    """
    # Create figure with subplots
    n_modalities = len([k for k in modalities.keys() if k != 'seg'])
    fig, axes = plt.subplots(1, n_modalities + 1, figsize=(20, 4))
    
    modality_names = ['t1', 't1ce', 't2', 'flair']
    
    # Display each modality
    for idx, modality in enumerate(modality_names):
        if modality in modalities:
            slice_img = extract_slice(modalities[modality], slice_idx, axis)
            axes[idx].imshow(slice_img, cmap='gray')
            axes[idx].set_title(f'{modality.upper()}', fontsize=14, fontweight='bold')
            axes[idx].axis('off')
    
    # Display segmentation if available
    if 'seg' in modalities:
        seg_slice = extract_slice(modalities['seg'], slice_idx, axis)
        axes[-1].imshow(seg_slice, cmap='jet', vmin=0, vmax=3)
        axes[-1].set_title('Segmentation', fontsize=14, fontweight='bold')
        axes[-1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def save_single_slice(slice_data, output_path, cmap='gray'):
    """
    Save a single slice as an image file.
    
    Args:
        slice_data: 2D numpy array
        output_path: Path to save the image
        cmap: Colormap to use (default: 'gray')
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_data, cmap=cmap)
    plt.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Slice saved to {output_path}")


# Example usage
if __name__ == "__main__":
    # Set your dataset path here
    dataset_path = "/path/to/BraTS2021_Training_Data"
    
    # Example: Load first case
    case_folder = "BraTS2021_00000"
    case_path = os.path.join(dataset_path, case_folder)
    
    if os.path.exists(case_path):
        # Load all modalities
        modalities = load_brats_case(case_path)
        
        # Visualize a slice from the middle of the brain (slice 77 is typical)
        visualize_all_modalities(modalities, slice_idx=77, axis=2, 
                                save_path='brats_slice_visualization.png')
        
        # Save individual slices
        if 't1' in modalities:
            t1_slice = extract_slice(modalities['t1'], slice_idx=77, axis=2)
            save_single_slice(t1_slice, 't1_slice_77.png')
        
        if 't1ce' in modalities:
            t1ce_slice = extract_slice(modalities['t1ce'], slice_idx=77, axis=2)
            save_single_slice(t1ce_slice, 't1ce_slice_77.png')
    else:
        print(f"Case path not found: {case_path}")
        print("\nTo use this script:")
        print("1. Download the BraTS 2021 dataset from Kaggle")
        print("2. Update the 'dataset_path' variable with your actual path")
        print("3. Update the 'case_folder' to match your data structure")


# Additional helper function to explore dataset structure
def explore_dataset(dataset_path, max_cases=5):
    """
    Explore the structure of BraTS dataset.
    
    Args:
        dataset_path: Path to the BraTS dataset
        max_cases: Maximum number of cases to display info for
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset path not found: {dataset_path}")
        return
    
    cases = sorted([d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"Total cases found: {len(cases)}")
    print(f"\nExploring first {min(max_cases, len(cases))} cases:\n")
    
    for case in cases[:max_cases]:
        case_path = os.path.join(dataset_path, case)
        files = os.listdir(case_path)
        print(f"Case: {case}")
        print(f"  Files: {files}")
        
        # Load one file to show dimensions
        nii_files = [f for f in files if f.endswith('.nii.gz')]
        if nii_files:
            sample_file = os.path.join(case_path, nii_files[0])
            img = nib.load(sample_file)
            print(f"  Sample shape: {img.shape}")
            print(f"  Data type: {img.get_data_dtype()}")
        print()