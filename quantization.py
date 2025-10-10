import numpy as np


def getQuantizationTable(algorithm):
    '''
    Inputs: 
        - algorithm (str): The (not case-sensitive) name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - quantization_table (8x8 np.ndarray): The quantization table for this algorithm
    '''
    pass


def quantize(matrices, algorithm):
    '''
    Inputs: 
        - matrices (nxmx8x8 np.ndarray): The matrices to be quantized, split into 8x8 arrays
        - algorithm (str): The (not case-sensitive) name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - new_matrices (nxmx8x8 np.ndarray): The matrices after having been quantized
    '''
    quantization_table = getQuantizationTable(algorithm.lower())
    
    shape = matrices.shape
    new_matrices = np.empty(shape, dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix = matrices[i, j]
            new_matrices[i, j] = np.rint(matrix/quantization_table)
    return new_matrices


def main():
    # Currently used for testing
    matrices = np.array(
        [[[1], [2], [8]],
         [[-3], [-1], [10]],
         [[0], [0], [1]]]
    )
    table = np.array(
        [5]
    )
    new_matrices = quantize(matrices, table)
    print(new_matrices)


if __name__ == '__main__':
    main()