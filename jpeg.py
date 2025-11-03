from loading import *
from dct import *
from quantization import *
from metrics import *
from utils import *
from JPEG_entropy import JPEG_encode
from JPEG_decode_lucie import JPEG_decode


def compressJPEG(original_image_data, Q):
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, Q)
    block_code, img_width, img_height, quantization_table = JPEG_encode(image_data, Q)
    return block_code, img_width, img_height, quantization_table


def decompressJPEG(block_code, img_width, img_height, quantization_table):
    compressed_image_data = JPEG_decode(block_code, img_width, img_height)
    image_data = decodeQuantization(compressed_image_data, quantization_table)
    image_data = decodeDCT(image_data)
    return image_data


def JPEG():
    CT_IMAGE = 123
    QUALITY = 50

    original_image_data = loadCT(CT_IMAGE)
    block_code, img_width, img_height, quantization_table = compressJPEG(original_image_data, QUALITY)
    decompressJPEG(block_code, img_width, img_height, quantization_table)

    # showMetrics(original_image_data, decompressed_image_data)

if __name__ == '__main__':
    JPEG()
