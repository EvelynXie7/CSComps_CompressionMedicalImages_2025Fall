import numpy as np


JPEG_LUMINANCE_QUANTIZATION_TABLE = np.array([ # From https://www.sciencedirect.com/topics/engineering/quantization-table
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def getQuantizationTable(algorithm):
    '''
    Inputs: 
        - algorithm (str): The name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - quantization_table (8x8 np.ndarray): The quantization table for this algorithm
    '''
    match algorithm:
        case 'jpeg':
            return JPEG_LUMINANCE_QUANTIZATION_TABLE


def quantize(unquantized_matrix, algorithm):
    '''
    Inputs: 
        - unquantized_matrix (nxmx8x8 np.ndarray): The un-quantized image data
        - algorithm (str): The name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - quantized_matrix (nxmx8x8 np.ndarray): The quantized image data
    '''
    quantization_table = getQuantizationTable(algorithm)

    shape = unquantized_matrix.shape
    quantized_matrix = np.empty(shape, dtype=np.int16)

    for i in range(shape[0]):
        for j in range(shape[1]):
            quantized_matrix[i, j] = np.rint(unquantized_matrix[i,j]/quantization_table)
    return quantized_matrix


def decodeQuantization(quantized_matrix, algorithm):
    '''
    Inputs: 
        - quantized_matrix (nxmx8x8 np.ndarray): The quantized image data
        - algorithm (str): The name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - unquantized_matrix (nxmx8x8 np.ndarray): The un-quantized image data
    '''
    quantization_table = getQuantizationTable(algorithm)

    shape = quantized_matrix.shape
    unquantized_matrix = np.empty(shape, dtype=np.int16)

    for i in range(shape[0]):
        for j in range(shape[1]):
            unquantized_matrix[i, j] = quantized_matrix[i,j] * quantization_table
    return unquantized_matrix
