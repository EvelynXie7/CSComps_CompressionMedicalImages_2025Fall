import numpy as np
import math
import os

def getMSE(image1, image2):
    '''
    Inputs: 
        - image1: the first image
        - image2: the second image
    Outputs:
        - mse (float): the Mean Squared Error between the two images
    '''
    image_1_retyped = image1.astype(np.int64)
    image_2_retyped = image2.astype(np.int64)

    if image_1_retyped.shape != image_2_retyped.shape:
        slices = tuple([slice(0, (min(image_1_retyped.shape[i], image_2_retyped.shape[i]))) for i in range(len(image_1_retyped.shape))])
        image_1_retyped = image_1_retyped[slices]
        image_2_retyped = image_2_retyped[slices]

    return np.mean((image_1_retyped - image_2_retyped) ** 2)

def getPSNR(image1, image2):
    '''
    Inputs: 
        - image1: the first image
        - image2: the second image
    Outputs:
        - psnr (float): the Peak Signal to Noise Ratio between the two images
    '''
    mse = getMSE(image1, image2)
    if mse != 0:
        return 10 * math.log((255 ** 2) / mse, 10)

def getPSNRjp2(image1, image2):
    '''
    Inputs: 
        - image1: the first image
        - image2: the second image
    Outputs:
        - psnr (float): the Peak Signal to Noise Ratio between the two images
    '''
    mse = getMSE(image1, image2)
    return 10 * math.log((65535 ** 2) / mse, 10)

def getCR(orig_image_size, compressed_filepath):
    '''
    Inputs: 
        - orig_image_size (int): the original image size, calculated before sending to this functin
        - compressed_filepath: the filepath to the compressed image, so that its size in bytes can be retrieved
    Outputs:
        - cr (float): the compression ratio, i.e. how much the image was compressed by
    '''
    compressed_bytes = os.path.getsize(compressed_filepath)
    return orig_image_size / compressed_bytes
