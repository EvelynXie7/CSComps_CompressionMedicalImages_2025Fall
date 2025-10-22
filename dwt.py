import pywt
import numpy as np

WAVELET = 'db1' # Chose Daubechiesfilter because that's what https://www.sciencedirect.com/science/article/pii/S2352914818302405 did
# Should probably change from 'db1' after doing more research

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
