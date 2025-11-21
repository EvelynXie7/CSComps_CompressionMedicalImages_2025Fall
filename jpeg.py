from loading import *
from dct import *
from quantization import *
from metrics import *
from utils import *
from JPEG_entropy import JPEG_encode
from JPEG_entropy_decode import JPEG_decode
import json
import time

def compressJPEG(original_image_data, Q, filedir, filename):
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, Q)
    JPEG_encode(image_data, Q, filedir, filename)


def decompressJPEG(comp_filedir, comp_filename, decomp_filedir, decomp_filename):
    '''
    Decompress a compressed JPEG stored in a binary file
    '''
    compressed_image_data, quantization_table = JPEG_decode(comp_filedir+'/'+comp_filename)
    image_data = decodeQuantization(compressed_image_data, quantization_table)
    image_data = decodeDCT(image_data)
    saveImage(image_data, decomp_filedir, decomp_filename)
    return image_data


def JPEG():
    '''
    Run the JPEG algorithm, encoding and decoding, on all KiTS data and BRaTS data. Save the binary file, 
    decompressed image, and MSE, PSNR, encoding_time, and decoding_time metrics. Uses a constant QUALITY=50
    '''
    QUALITY = 50

    for case in range(300):
        print('Case:', case)
        original_case = loadCT(case)

        comp_filedir = getFilepath('compressed_data', 'JPEG', case)
        if not os.path.exists(comp_filedir): 
            os.makedirs(comp_filedir)
        
        decomp_filedir = getFilepath('decompressed_data', 'JPEG', case)
        if not os.path.exists(decomp_filedir): 
            os.makedirs(decomp_filedir)
        
        metrics_filedir = getFilepath('metric_data', 'JPEG', case)
        if not os.path.exists(metrics_filedir): 
            os.makedirs(metrics_filedir)

        for slice in range(original_case.shape[0]):
            original_slice = original_case[slice]

            comp_filename = getFilename(slice, 'bin')
            decomp_filename = getFilename(slice, 'png')
            metrics_filename = getFilename(slice, 'json')

            start_compress_time = time.time()
            compressJPEG(original_slice, QUALITY, comp_filedir, comp_filename)
            start_decompress_time = time.time()
            decomp_slice = decompressJPEG(QUALITY, comp_filedir, comp_filename, decomp_filedir, decomp_filename)
            end_time = time.time()

            metrics = {
                'MSE': getMSE(original_slice, decomp_slice),
                'PSNR': getPSNR(original_slice, decomp_slice),
                'encoding_time': start_decompress_time - start_compress_time,
                'decoding_time': end_time - start_decompress_time
            }
            
            with open(metrics_filedir+'/'+metrics_filename, 'w') as f:
                json.dump(metrics, f)
        

if __name__ == '__main__':
    JPEG()
