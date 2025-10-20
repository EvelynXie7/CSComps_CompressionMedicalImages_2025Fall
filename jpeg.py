from loading import *
from dct import *
from quantization import *
from metrics import *
from utils import *


def compressJPEG(original_image_data):
    compressed_image_data = runDCT(original_image_data)
    compressed_image_data = quantize(compressed_image_data, 'jpeg')
    return compressed_image_data


def decompressJPEG(compressed_image_data):
    decompressed_image_data = decodeQuantization(compressed_image_data, 'jpeg')
    decompressed_image_data = decodeDCT(decompressed_image_data)
    return decompressed_image_data


def JPEG():
    original_image_data = loadCT(123, save=True)
    compressed_image_data = compressJPEG(original_image_data)
    decompressed_image_data = decompressJPEG(compressed_image_data)

    saveImage(decompressed_image_data, 'ct_jpeg')
    showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    JPEG()
