from scipy.fft import dct, idct
import numpy as np

def reshapeImageForDCT(image_data):
    '''
    Inputs: 
        - image_data (nxm np.ndarray): The data for a grayscale image, where each item is the grayscale of that pixel
    Outputs:
        - reshaped ((n/8)x(m/8)x8x8 np.ndarray): The same image, but with its data split into 8x8 chunks, 
          and instead of values from 0 to 255, -128 to 127
    '''
    h, w = image_data.shape
    reshaped = np.empty((h // 8, w // 8, 8, 8), dtype=np.int16)

    for i in range(h // 8):
        for j in range(w // 8):
            reshaped[i,j] = image_data[8*i:8*i+8, 8*j:8*j+8]

    return reshaped - 128


def reshapeImageFromDCT(image_data):
    '''
    Inputs: 
        - image_data (nxmx8x8 np.ndarray): Data for an image, split into 8x8 chunks of pixels
    Outputs:
        - reshaped ((n*8)x(m*8) np.ndarray): Reshaped data for the image, returning it from 8x8 chunks to regular form
    '''
    h, w, _, _ = image_data.shape
    reshaped = np.empty((h * 8, w * 8), dtype=np.int16)

    for i in range(h):
        for j in range(w):
            reshaped[8*i:8*i+8, 8*j:8*j+8] = image_data[i, j]

    return reshaped + 128


def runDCT(image_data):
    matrices = reshapeImageForDCT(image_data)
    matrices  = dct(dct(matrices.T,  norm='ortho').T, norm='ortho')
    return matrices


def decodeDCT(matrices):
    image_data  = idct(idct(matrices.T,  norm='ortho').T, norm='ortho')
    image_data = reshapeImageFromDCT(image_data)
    image_data[image_data > 255] = 255
    image_data[image_data < 0] = 0
    return image_data
