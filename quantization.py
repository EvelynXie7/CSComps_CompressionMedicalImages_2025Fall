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

JPEG_CHROMINANCE_QUANTIZATION_TABLE = np.array([ # From https://www.sciencedirect.com/topics/engineering/quantization-table
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def getQuantizationTable(algorithm, component):
    '''
    Inputs: 
        - algorithm (str): The name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
        - component (str): The name of the component being quantized, i.e. "y" or "cr"
    Outputs:
        - quantization_table (8x8 np.ndarray): The quantization table for this algorithm
    '''
    match algorithm:
        case 'jpeg':
            if component.lower() == 'y':
                return JPEG_LUMINANCE_QUANTIZATION_TABLE
            else:
                return JPEG_CHROMINANCE_QUANTIZATION_TABLE


def quantizeComponent(matrix, algorithm, component_name):
    quantization_table = getQuantizationTable(algorithm, component_name)

    shape = matrix.shape
    new_matrix = np.empty(shape, dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            new_matrix[i, j] = np.rint(matrix[i,j]/quantization_table)
    return new_matrix


def quantize(matrices, algorithm):
    y_orig, cb_orig, cr_orig = matrices
    
    y_quantized =  quantizeComponent(y_orig,  algorithm, 'y')
    cb_quantized = quantizeComponent(cb_orig, algorithm, 'cb')
    cr_quantized = quantizeComponent(cr_orig, algorithm, 'cr')
        
    return (y_quantized, cb_quantized, cr_quantized)


def decodeQuantizeComponent(matrix, algorithm, component_name):
    quantization_table = getQuantizationTable(algorithm, component_name)

    shape = matrix.shape
    new_matrix = np.empty(shape, dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            new_matrix[i, j] = np.rint(matrix[i,j] * quantization_table)
    return new_matrix


def decodeQuantization(quantized_matrices, algorithm):
    y_orig, cb_orig, cr_orig = quantized_matrices
    
    y_unquantized =  decodeQuantizeComponent(y_orig,  algorithm, 'y')
    cb_unquantized = decodeQuantizeComponent(cb_orig, algorithm, 'cb')
    cr_unquantized = decodeQuantizeComponent(cr_orig, algorithm, 'cr')
        
    return (y_unquantized, cb_unquantized, cr_unquantized)


def test():
    matrices = np.array(
        [[[1], [2], [8]],
         [[-3], [-1], [10]],
         [[0], [0], [1]]]
    )
    new_matrices = quantize(matrices, 'test')
    matrices = decodeQuantization(new_matrices, 'test')
    print(matrices)

if __name__ == '__main__':
    test()
