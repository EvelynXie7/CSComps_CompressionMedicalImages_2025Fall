from loading import *
from dct import *
from quantization import *
from metrics import *
from utils import *
from JPEG_entropy import JPEG_encode


def compressJPEG(original_image_data, Q):
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, Q)
    JPEG_encode(image_data, Q)
    return image_data


def decompressJPEG(compressed_image_data, Q):
    image_data = decodeQuantization(compressed_image_data, Q)
    image_data = decodeDCT(image_data)
    return image_data


def JPEG():
    CT_IMAGE = 123
    QUALITY = 50

    original_image_data = loadCT(CT_IMAGE)
    compressed_image_data = compressJPEG(original_image_data, QUALITY)

    # showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    JPEG()
