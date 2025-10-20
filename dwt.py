import pywt
import numpy as np

WAVELET = 'db1' # Chose Daubechiesfilter because that's what https://www.sciencedirect.com/science/article/pii/S2352914818302405 did

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


def runDWT(image_data):
    LL, (LH, HL, HH) = pywt.dwt2(image_data, WAVELET)
    # LL_LL, (LL_LH, LL_HL, LL_HH) = pywt.dwt2(LL, WAVELET)
    # LL_LL_LL, (LL_LL_LH, LL_LL_HL, LL_LL_HH) = pywt.dwt2(LL_LL, WAVELET)

    # LL_LL = combineDWTData(LL_LL_LL, LL_LL_LH, LL_LL_HL, LL_LL_HH)
    # LL = combineDWTData(LL_LL, LL_LH, LL_HL, LL_HH)
    dwt_data = combineDWTData(LL, LH, HL, HH)

    return dwt_data

def decodeDWT(dwt_data):
    LL, LH, HL, HH = separateDWTData(dwt_data)
    # LL_LL, LL_LH, LL_HL, LL_HH = separateDWTData(LL)
    # LL_LL_LL, LL_LL_LH, LL_LL_HL, LL_LL_HH = separateDWTData(LL_LL)

    # LL_LL = pywt.idwt2((LL_LL_LL, (LL_LL_LH, LL_LL_HL, LL_LL_HH)), WAVELET)
    # LL = pywt.idwt2((LL_LL, (LL_LH, LL_HL, LL_HH)), WAVELET)
    image_data = pywt.idwt2((LL, (LH, HL, HH)), WAVELET)

    return image_data