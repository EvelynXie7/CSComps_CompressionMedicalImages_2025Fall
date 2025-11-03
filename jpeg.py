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
    CASE_NUM = 123
    SLICE_NUM = 0
    QUALITY = 50

    original_image_data = loadCT(CASE_NUM, SLICE_NUM)
    compressJPEG(original_image_data, QUALITY)
    decompressed_image = decompressJPEG()

if __name__ == '__main__':
    JPEG()
