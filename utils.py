import numpy as np
from PIL import Image
import os


def saveImage(image_data, filedir, filename):
    PIL_image = Image.fromarray(image_data.astype(np.uint8))
    PIL_image = PIL_image.convert('RGB')
    PIL_image.save(filedir+'/'+filename)


def numToStr(type, val):
    val_str = str(val)
    if len(val_str) == 1:
        val_str = '00' + val_str
    elif len(val_str) == 2:
        val_str = '0' + val_str
    return f'{type.lower()}_{val_str}'


def getFilepath(type, algorithm, case):
    case_dir = numToStr('case', case)
    return f'outputs/{type}/{algorithm.upper()}/{case_dir}'

def getFilename(slice, ext):
    slice_str = numToStr('slice', slice)
    return f'{slice_str}.{ext}'