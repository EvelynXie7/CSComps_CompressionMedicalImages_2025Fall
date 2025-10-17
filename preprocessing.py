'''
CT Data and functions to load images from https://github.com/neheller/kits19#
'''

import numpy as np
import nibabel as nib

from starter_code.utils import get_case_path
from starter_code.visualize import hu_to_grayscale, DEFAULT_HU_MIN, DEFAULT_HU_MAX
import utils


def loadCT(cid, save=False):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    
    vol_data = vol.get_fdata()
    vol_data = hu_to_grayscale(vol_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)[0, :, :, 0].astype(np.uint8) # 0 to get first image in the structure

    if save:
        utils.saveCTImage(vol_data, name='original_ct')

    return vol_data.astype(np.int16)


def loadMRI():
    pass
