import numpy as np
from PIL import Image

DEFAULT_HU_MAX = 512
DEFAULT_HU_MIN = -512


def saveImage(image_data, filename):
    '''
    Saves an image to a file given an array of grayscale pixel data.

    Inputs: 
        image_data - the image data to save as an image
        filename - the location to save the image in
    Outputs:
        Saves the file.
    '''
    PIL_image = Image.fromarray(image_data.astype(np.uint8))
    PIL_image = PIL_image.convert('RGB')
    PIL_image.save(filename)
