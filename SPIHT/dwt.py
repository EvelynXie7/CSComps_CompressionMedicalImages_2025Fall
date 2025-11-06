import pywt
import numpy as np

WAVELET = 'db1'

def combineDWTData(LL, LH, HL, HH):
    """
    Arrange DWT subbands to match SPIHT paper's expected layout
    Places LL in top-left, with HL, LH, HH surrounding it
    This creates quad-tree structure for SPIHT
    """
    h, w = LL.shape
    
    # Create output array (2h × 2w)
    result = np.zeros((2*h, 2*w), dtype=LL.dtype)
    
    # Place subbands in quad-tree layout
    result[0:h, 0:w] = LL      # Top-left
    result[0:h, w:2*w] = HL    # Top-right
    result[h:2*h, 0:w] = LH    # Bottom-left
    result[h:2*h, w:2*w] = HH  # Bottom-right
    
    return result


def separateDWTData(dwt_data):
    """
    Extract subbands from SPIHT-style layout
    """
    h, w = dwt_data.shape
    h_half = h // 2
    w_half = w // 2

    LL = dwt_data[0:h_half, 0:w_half]
    HL = dwt_data[0:h_half, w_half:w]
    LH = dwt_data[h_half:h, 0:w_half]
    HH = dwt_data[h_half:h, w_half:w]

    return LL, HL, LH, HH


def runDWT(image_data, level):
    """Recursive DWT with SPIHT-compatible layout"""
    if level == 0:
        return image_data
    
    LL, (LH, HL, HH) = pywt.dwt2(image_data, WAVELET)
    
    # Recursively decompose LL
    LL_after_DWT = runDWT(LL, level - 1)
    
    # Combine in SPIHT layout
    dwt_data = combineDWTData(LL_after_DWT, LH, HL, HH)

    return dwt_data


def decodeDWT(dwt_data, level):
    """Recursive inverse DWT"""
    if level == 0:
        return dwt_data
    
    LL, LH, HL, HH = separateDWTData(dwt_data)
    
    # Recursively reconstruct LL
    LL_after_IDWT = decodeDWT(LL, level - 1)
    
    # Inverse DWT
    image_data = pywt.idwt2((LL_after_IDWT, (LH, HL, HH)), WAVELET)

    return image_data
