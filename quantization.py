import numpy as np


def quantizeFromTable(matrices, quantization_table):
    '''
    Inputs: 
        - matrices (nxmx8x8 np.ndarray): The matrices to be quantized, split into 8x8 arrays, 
          n and m relate to the dimensions of the original image
        - quantization_table (8x8 np.ndarray): The 8x8 quantization table
    Outputs:
        - new_matrices (nxmx8x8 np.ndarray): The matrices after having been quantized
    '''
    shape = matrices.shape
    new_matrices = np.empty(shape, dtype=int)

    for i in range(shape[0]):
        for j in range(shape[1]):
            matrix = matrices[i, j]
            new_matrices[i, j] = np.rint(matrix/quantization_table)
    return new_matrices


def quantize(matrices, algorithm): # Function will work by finding the correct quantization table to use, then calling quantizeFromTable()
    '''
    Inputs: 
        - matrices (nxmx8x8 np.ndarray): The matrices to be quantized, split into 8x8 arrays
        - algorithm (str): The (not case-sensitive) name of the algorithm being quantized, i.e. "jpeg" or "jpeg2000"
    Outputs:
        - new_matrices (nxmx8x8 np.ndarray): The matrices after having been quantized
    '''
    pass


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