import numpy as np
from PIL import Image
import os

DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512


def saveImage(image_data, filename):
    PIL_image = Image.fromarray(image_data.astype(np.uint8))
    PIL_image = PIL_image.convert('RGB')
    PIL_image.save(filename)


def numToStr(type, val):
    val_str = str(val)
    if len(val_str) == 1:
        val_str = '00' + val_str
    elif len(val_str) == 2:
        val_str = '0' + val_str
    return f'{type.lower()}_{val_str}'


def getFilename(slice, ext):
    slice_str = numToStr('slice', slice)
    return f'{slice_str}.{ext}'