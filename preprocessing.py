'''
CT Data and functions to load images from https://github.com/neheller/kits19#
'''

from starter_code.utils import *
from starter_code.visualize import *
import numpy as np
import nibabel as nib
from pathlib import Path
from imageio import imwrite
from PIL import Image

def loadCT(cid, img_num=0):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))

    vol_data = vol.get_fdata()
    vol_data = hu_to_grayscale(vol_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX).astype(np.uint8)
    return vol_data[img_num]
    
def loadMRI():
    pass

def runDCT(image):
    pass

def runDWT(image):
    pass

def saveImg(img, name='current_image'):
    out_path = Path('tmp/images/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    imwrite(str(fpath), img)

if __name__ == '__main__':
    vol_data = loadCT(123, 10)
    saveImg(vol_data)