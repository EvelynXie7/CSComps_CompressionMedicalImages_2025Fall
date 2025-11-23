import pywt
from scipy.fft import dct, idct
import numpy as np

WAVELET =  pywt.Wavelet('db1') # Chose Daubechiesfilter because that's what https://www.sciencedirect.com/science/article/pii/S2352914818302405 did

def combineDWTData(LL, LH, HL, HH):
    L_data = np.vstack((LL, LH))
    H_data = np.vstack((HL, HH))
    return np.hstack((L_data, H_data))


def separateDWTData(dwt_data):
    x_len = dwt_data.shape[0] // 2
    y_len = dwt_data.shape[1] // 2

    LL = dwt_data[:x_len, :y_len]
    LH = dwt_data[:x_len, y_len:]
    HL = dwt_data[x_len:, :y_len]
    HH = dwt_data[x_len:, y_len:]

    return LL, LH, HL, HH


def runDWT(image_data, level):
    if level == 0:
        return image_data
    
    LL, (LH, HL, HH) = pywt.dwt2(image_data, WAVELET)
    LL_after_DWT = runDWT(LL, level - 1)
    dwt_data = combineDWTData(LL_after_DWT, LH, HL, HH)

    return dwt_data


def decodeDWT(dwt_data, level):
    if level == 0:
        return dwt_data
    
    LL, LH, HL, HH = separateDWTData(dwt_data)
    LL_after_IDWT = decodeDWT(LL, level - 1)
    image_data = pywt.idwt2((LL_after_IDWT, (LH, HL, HH)), WAVELET)

    return image_data

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

    resized = image_data
    new_h = h
    new_w = w

    if h_remainder != 0:
        new_h = h+8-h_remainder
    if w_remainder != 0:
        new_w = w+8-w_remainder
    
    if h_remainder != 0 or w_remainder != 0:
        resized = np.empty((new_h, new_w), dtype=np.int16)
        resized[:h, :w] = image_data[:, :]
        for i in range(h, new_h):
            resized[i, :w] = image_data[-1, :]
        for i in range(w, new_w):
            resized[:h, i] = image_data[:, -1]
        resized[h:, w:].fill(image_data[-1, -1])

    
    reshaped = np.empty((new_h // 8, new_w // 8, 8, 8), dtype=np.int16)

    for i in range(new_h // 8):
        for j in range(new_w // 8):
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
