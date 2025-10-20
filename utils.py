import numpy as np
from pathlib import Path
from imageio import imwrite
from PIL import Image


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


def saveImage(image_data, name='current_image'):
    image = toPILImage(image_data)
    savePILImage(image, name)