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
    h_remainder = h % 8
    w_remainder = h % 8

    if h_remainder != 0:
        h += 8 - (h_remainder)
    if w_remainder != 0:
        w += 8 - (w_remainder)

    resized = image_data
    if h_remainder != 0 or w_remainder != 0:
        resized = np.empty((h, w), dtype=np.int16)
        resized[:(h-h_remainder), :(w-w_remainder)] = image_data[:, :]
        for i in range(h-h_remainder, h):
            resized[i, :(w-w_remainder)] = image_data[-1, :]
        for i in range(w-w_remainder, w):
            resized[:(h-h_remainder), i] = image_data[:, -1]
        resized[(h-h_remainder):, (w-w_remainder):].fill(image_data[:, -1])

    
    reshaped = np.empty((h // 8, w // 8, 8, 8), dtype=np.int16)

    for i in range(h // 8):
        for j in range(w // 8):
            reshaped[i,j] = resized[8*i:8*i+8, 8*j:8*j+8]


    return reshaped - 128


def reshapeImageFromDCT(matrices):
    '''
    Inputs: 
        - image_data (nxmx8x8 np.ndarray): Data for an image, split into 8x8 chunks of pixels
    Outputs:
        - reshaped ((n*8)x(m*8) np.ndarray): Reshaped data for the image, returning it from 8x8 chunks to regular form
    '''
    h, w, _, _ = matrices.shape
    reshaped = np.empty((h * 8, w * 8), dtype=np.int16)

    for i in range(h):
        for j in range(w):
            reshaped[8*i:8*i+8, 8*j:8*j+8] = matrices[i, j]

    return reshaped + 128


def runDCT(image_data):
    matrices = reshapeImageForDCT(image_data)
    h, w = matrices.shape[0], matrices.shape[1]
    for i in range(h):
        for j in range(w):
            matrices[i, j] = dct(dct(matrices[i, j], axis=0, norm='ortho'), axis=1, norm='ortho')
    return matrices


def decodeDCT(matrices):
    h, w = matrices.shape[0], matrices.shape[1]
    for i in range(h):
        for j in range(w):
            matrices[i, j] = idct(idct(matrices[i, j], axis=1, norm='ortho'), axis=0, norm='ortho')
    image_data = reshapeImageFromDCT(matrices)
    image_data[image_data > 255] = 255
    image_data[image_data < 0] = 0
    return image_data
