from pywt import dwt2, idwt2
from utils import *

def runDWT(image_data):
    matrices = reshapeImage(image_data)
    matrices = dwt2(matrices)
    return matrices


def decodeDWT(matrices):
    matrices = reshapeImage(matrices)
    image_data = idwt2(matrices)
    return image_data
