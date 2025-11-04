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


def decompressJPEG(filename="output.bin", Q=50):
    compressed_image_data = JPEG_decode(filename)
    quantization_table = getJPEGQuantizationTable(Q)
    image_data = decodeQuantization(compressed_image_data, quantization_table)
    image_data = decodeDCT(image_data)
    return compressed_image_data, image_data


def JPEG():
    CASE_NUM = 123
    SLICE_NUM = 0
    QUALITY = 50

    original_image_data = loadCT(CASE_NUM, SLICE_NUM)
    compressJPEG(original_image_data, QUALITY)
    compressed_image, decompressed_image = decompressJPEG()
    #maybe add visualization with mat plot lib to see if works

if __name__ == '__main__':
    JPEG()
