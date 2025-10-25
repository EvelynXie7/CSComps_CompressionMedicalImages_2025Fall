import numpy as np
from pathlib import Path
from imageio import imwrite
from PIL import Image


def saveImage(image_data, name='current_image'):
    PIL_image = Image.fromarray(image_data.astype(np.uint8))
    PIL_image = PIL_image.convert('RGB')
    
    out_path = Path('tmp/')
    if not out_path.exists():
        out_path.mkdir()  

    fpath = out_path / f'{name}.png'
    PIL_image.save(fpath)
