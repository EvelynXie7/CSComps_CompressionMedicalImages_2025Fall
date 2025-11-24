import numpy as np


JPEG_LUMINANCE_QUANTIZATION_TABLE = np.array([ 
    # From https://www.sciencedirect.com/topics/engineering/quantization-table
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])


def getJPEGQuantizationTable(Q):
    '''
    Create the JPEG Quantization table given a quality Q.

    Input:
        Q - a quality parameter that is an integer between 1 and 99 inclusive, where 1 is 
            high quality and 99 higher compression
    
    Output:
        table - the new quantization table
    '''
    assert Q >= 1 and Q <= 99

    table = JPEG_LUMINANCE_QUANTIZATION_TABLE

    if Q == 50:
        return table
    elif Q > 50:
        table = table * int((100-Q) / 50)
        table[table < 1] = 1
        return table
    else:
        return table * int(50 / Q)



def quantize(unquantized_matrix, Q):
    '''
    Quantizes all chunks in the image data, dividing all values in an 8x8 chunk by their corresponding
    value in the correct quantization table and rounded to the nearest integer.

    Inputs: 
        unquantized_matrix - The un-quantized image data
        Q - The quality of the compression/quantization, to be used when creating the quantization table
    Outputs:
        quantized_matrix - The quantized image data
    '''
    quantization_table = getJPEGQuantizationTable(Q)

    shape = unquantized_matrix.shape
    quantized_matrix = np.empty(shape, dtype=np.int16)

    for i in range(shape[0]):
        for j in range(shape[1]):
            quantized_matrix[i, j] = np.rint(unquantized_matrix[i,j] / quantization_table)
    return quantized_matrix


def decodeQuantization(quantized_matrix, quantization_table):
    '''
    Un-quantizes all chunks in the image data, multiplying all values in an 8x8 chunk by their corresponding
    value in the given quantization table.

    Inputs: 
        quantized_matrix - the image data after quantization
        quantization_table - the quantization table used when originally quantizing the data
    Outputs:
        unquantized_matrix: the unquantized data; the data after being multiplied back
    '''
    shape = quantized_matrix.shape
    unquantized_matrix = np.empty(shape, dtype=np.int16)

    for i in range(shape[0]):
        for j in range(shape[1]):
            unquantized_matrix[i, j] = quantized_matrix[i,j] * quantization_table
    return unquantized_matrix
