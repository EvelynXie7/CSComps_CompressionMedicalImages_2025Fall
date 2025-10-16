from scipy.fft import dct, idct
import numpy as np


def reshapeImage(image_data):
    h, w = image_data.shape

    reshaped = np.empty((h // 8, w // 8, 8, 8), dtype=np.int16)

    for i in range(h // 8):
        for j in range(w // 8):
            reshaped[i,j] = image_data[8*i:8*i+8, 8*j:8*j+8]

    return reshaped


def decodeReshapeImage(image_data):
    h, w, _, _ = image_data.shape

    reshaped = np.empty((h * 8, w * 8), dtype=np.int16)

    for i in range(h):
        for j in range(w):
            reshaped[8*i:8*i+8, 8*j:8*j+8] = image_data[i, j]

    return reshaped


def runDCT(image_data):
    reshaped = reshapeImage(image_data - 128)
    reshaped  = dct(dct(reshaped.T,  norm='ortho').T, norm='ortho')
    return reshaped
    
    
def decodeDCT(matrices):
    image_data  = idct(idct(matrices.T,  norm='ortho').T, norm='ortho')
    image_data = decodeReshapeImage(image_data) + 128
    image_data[image_data > 255] = 255
    image_data[image_data < 0] = 0
    return image_data
    
