import numpy as np
import math

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
    return 10 * math.log((255 ** 2) / mse, 10)

def getCR(compressed_image, orig_image):
    '''
    Inputs: 
        - compressed_image: the image/data needed to recreate the image post-compression
        - orig_image: the original image
    Outputs:
        - cr (float): the compression ratio, i.e. how much the image was compressed by
    '''
    pass

def showMetrics(image1, image2):
    mse = getMSE(image1, image2)
    psnr = getPSNR(image1, image2)

    print('\nCompression statistics:\n-----------')
    print('MSE: ', round(mse, 5))
    print('PSNR:', round(psnr, 5))
    print()
