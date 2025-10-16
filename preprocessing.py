'''
CT Data and functions to load images from https://github.com/neheller/kits19#
'''

import numpy as np
import nibabel as nib
from pathlib import Path
from imageio import imwrite
from PIL import Image

from ct_preprocessing_starter_code.utils import get_case_path
from ct_preprocessing_starter_code.visualize import hu_to_grayscale, DEFAULT_HU_MIN, DEFAULT_HU_MAX


def loadCT(cid, save=False):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    
    vol_data = vol.get_fdata()
    vol_data = hu_to_grayscale(vol_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)[0, :, :, 0].astype(np.uint8) # 0 to get first image in the structure

    if save:
        saveCTImage(vol_data, name='original_ct')

    return vol_data.astype(np.int16)


def loadMRI():
    pass

def runDWT(image_data):
    pass

def toPILImage(image_data):
    PIL_image = Image.fromarray(image_data.astype(np.uint8))
    PIL_image = PIL_image.convert('RGB')
    return PIL_image


def savePILImage(image, name='current_image'):
    out_path = Path('tmp/images/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    image.save(fpath)


def saveCTImage(image, name='current_image'):
    out_path = Path('tmp/images/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    imwrite(str(fpath), image)