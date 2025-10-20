from loading import *
from dwt import *
from metrics import *
from utils import *

def compressSPIHT(original_image_data):
    compressed_image_data = runDWT(original_image_data)
    return compressed_image_data

def decompressSPIHT(compressed_image_data):
    decompressed_image_data = decodeDWT(compressed_image_data)
    return decompressed_image_data

def SPIHT():
    original_image_data = loadCT(123, save=False)
    compressed_image_data = compressSPIHT(original_image_data)
    decompressed_image_data = decompressSPIHT(compressed_image_data)

    saveImage(decompressed_image_data, 'ct_spiht')
    showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    SPIHT()
