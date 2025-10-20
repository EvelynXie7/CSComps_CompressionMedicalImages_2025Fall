from scipy.fft import dct, idct
from utils import reshapeImageForDCT, reshapeImageFromDCT

def runDCT(image_data):
    matrices = reshapeImageForDCT(image_data)
    matrices  = dct(dct(matrices.T,  norm='ortho').T, norm='ortho')
    return matrices

def decodeDCT(matrices):
    image_data  = idct(idct(matrices.T,  norm='ortho').T, norm='ortho')
    image_data = reshapeImageFromDCT(image_data)
    image_data[image_data > 255] = 255
    image_data[image_data < 0] = 0
    return image_data
