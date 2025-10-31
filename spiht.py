from loading import *
from dwt import *
from metrics import *
from utils import *

def compressSPIHT(original_image_data, level):
    compressed_image_data = runDWT(original_image_data, level)
    return compressed_image_data

def decompressSPIHT(compressed_image_data, level):
    decompressed_image_data = decodeDWT(compressed_image_data, level)
    return decompressed_image_data

def SPIHT():
    CT_IMAGE = 123

    original_image_data = loadCT(CT_IMAGE)
    compressed_image_data = compressSPIHT(original_image_data, level=3)
    decompressed_image_data = decompressSPIHT(compressed_image_data, level=3)

    saveImage(decompressed_image_data, 'ct_spiht')
    showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    SPIHT()
