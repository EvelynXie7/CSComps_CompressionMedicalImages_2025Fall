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

from scipy.fft import dct, idct


def loadCT(cid, image_num=0):
    case_path = get_case_path(cid)
    vol = nib.load(str(case_path / "imaging.nii.gz"))
    

    vol_data = vol.get_fdata()
    vol_data = hu_to_grayscale(vol_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX).astype(np.uint8)

    saveIOImage(vol_data[0], 'original')

    image = Image.fromarray(vol_data[image_num]) # Cutting down on data here rather than before for best scaling data during hu_to_grayscale
    image_data = np.array(image.convert('YCbCr')).astype(np.int16)
    return image_data
    

def loadMRI():
    pass


def reshapeImage(image_data):
    h, w, _ = image_data.shape

    reshaped = np.empty((h // 8, w // 8, 8, 8, 3), dtype=int)

    for i in range(h // 8):
        for j in range(w // 8):
            reshaped[i,j] = image_data[8*i:8*i+8, 8*j:8*j+8]

    y = reshaped[:, :, :, :, 0]
    cb = reshaped[:, :, :, :, 1]
    cr = reshaped[:, :, :, :, 2]

    return y, cb, cr

def decodeReshapeImage(image_data):
    y, cb, cr = image_data
    h, w, _, _ = y.shape

    reshaped = np.empty((h * 8, w * 8, 3), dtype=int)

    for i in range(h):
        for j in range(w):
            reshaped[8*i:8*i+8, 8*j:8*j+8, 0] = y[i, j, :, :]
            reshaped[8*i:8*i+8, 8*j:8*j+8, 1] = cb[i, j, :, :]
            reshaped[8*i:8*i+8, 8*j:8*j+8, 2] = cr[i, j, :, :]

    return reshaped



def runDCT(image_data):
    y, cb, cr = reshapeImage(image_data - 128)

    y_dct  = dct(dct(y.T,  norm='ortho').T, norm='ortho')
    cb_dct = dct(dct(cb.T, norm='ortho').T, norm='ortho')
    cr_dct = dct(dct(cr.T, norm='ortho').T, norm='ortho')

    return y_dct, cb_dct, cr_dct
    
def decodeDCT(matrices):
    y, cb, cr = matrices

    y_idct  = idct(idct(y.T,  norm='ortho').T, norm='ortho')
    cb_idct = idct(idct(cb.T, norm='ortho').T, norm='ortho')
    cr_idct = idct(idct(cr.T, norm='ortho').T, norm='ortho')

    image = decodeReshapeImage([y_idct, cb_idct, cr_idct]) + 128
    return image
    

def runDWT(image_data):
    pass

def toPILImage(image_data):
    PIL_image = Image.fromarray(image_data, mode='YCbCr')
    PIL_image = PIL_image.convert('RGB')
    return PIL_image


def saveImage(image, name='current_image'):
    out_path = Path('tmp/images/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    image.save(fpath)


def saveIOImage(image, name='current_image'):
    out_path = Path('tmp/images/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    imwrite(str(fpath), image)