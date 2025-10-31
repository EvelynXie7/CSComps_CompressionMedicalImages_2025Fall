'''
CT Data and functions to load images from https://github.com/neheller/kits19#
'''

import numpy as np
import nibabel as nib

from kits_utils import get_case_path
from kits_visualize import hu_to_grayscale, DEFAULT_HU_MIN, DEFAULT_HU_MAX
import utils


def loadCT(cid):
    IMAGE_IN_STRUCTURE = 0

    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    seg = nib.load(str(case_path / "segmentation.nii.gz"))
    
    vol_data = vol.get_fdata()
    vol_data = hu_to_grayscale(vol_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)[IMAGE_IN_STRUCTURE, :, :, 0]
    seg_data = seg.get_fdata()[IMAGE_IN_STRUCTURE]

    return vol_data

def loadMRI():
    # Set up when have access to other computer to confirm there
    pass
