from loading import *
from dct import *
from quantization import *
from metrics import *
from utils import *
from JPEG_entropy import JPEG_encode
from JPEG_entropy_decode import JPEG_decode


def compressJPEG(original_image_data, Q):
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, Q)
    JPEG_encode(image_data, Q)


def decompressJPEG():
    compressed_image_data, quantization_table = JPEG_decode()
    image_data = decodeQuantization(compressed_image_data, quantization_table)
    image_data = decodeDCT(image_data)
    return image_data


def JPEG():
    CT_IMAGE = 123
    QUALITY = 80

    original_image_data = loadCT(CT_IMAGE)
    compressJPEG(original_image_data, QUALITY)
    decompressed_image = decompressJPEG()

    saveImage(decompressed_image, 'ct_jpeg_decoded')
    # showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    JPEG()
