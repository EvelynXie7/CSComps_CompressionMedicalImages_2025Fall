import nibabel as nib
import json
import time

from dct import *
from quantization import *
from metrics import *
from utils import *
from JPEG_entropy import JPEG_encode
from JPEG_entropy_decode import JPEG_decode


def compressJPEG(original_image_data, Q, filename):
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, Q)
    JPEG_encode(image_data, Q, filename)


def decompressJPEG(comp_filename, decomp_filename):
    '''
    Decompress a compressed JPEG stored in a binary file
    '''
    compressed_image_data, quantization_table = JPEG_decode(comp_filename)
    image_data = decodeQuantization(compressed_image_data, quantization_table)
    image_data = decodeDCT(image_data)
    saveImage(image_data, decomp_filename)
    return image_data


def runJPEGOnCase(case_data, case_id, output_dir, quality, slice_axis):
    max_val = np.max(case_data)
    min_val = np.min(case_data)
    if max_val > min_val:
        case_data = 255 * (case_data - min_val) / (max_val - min_val)
    else:
        case_data = np.zeros_like(case_data)
    case_data = case_data.astype(np.uint8)

    comp_filedir = os.path.join(output_dir, 'compressed_data', 'JPEG', case_id)
    decomp_filedir = os.path.join(output_dir, 'decompressed_data', 'JPEG', case_id)
    metrics_filedir = os.path.join(output_dir, 'metric_data', 'JPEG', case_id)

    os.makedirs(comp_filedir, exist_ok=True)
    os.makedirs(decomp_filedir, exist_ok=True)
    os.makedirs(metrics_filedir, exist_ok=True)

    for slice_num in range(case_data.shape[slice_axis]):
        slice_id = "slice_{:05d}".format(slice_num)
        slice_data = None
        match slice_axis:
            case 0:
                slice_data = case_data[slice_num, :, :]
            case 1:
                slice_data = case_data[:, slice_num, :]
            case 2:
                slice_data = case_data[:, :, slice_num]
            case _:
                print("Invalid slice.")
                exit()

        comp_filename = os.path.join(comp_filedir, f'{slice_id}.bin')
        decomp_filename = os.path.join(decomp_filedir, f'{slice_id}.png')
        metrics_filename = os.path.join(metrics_filedir, f'{slice_id}.json')

        start_compress_time = time.time()
        compressJPEG(slice_data, quality, comp_filename)
        start_decompress_time = time.time()
        decomp_slice = decompressJPEG(comp_filename, decomp_filename)
        end_time = time.time()

        metrics = {
            'MSE': getMSE(slice_data, decomp_slice),
            'PSNR': getPSNR(slice_data, decomp_slice),
            'encoding_time': start_decompress_time - start_compress_time,
            'decoding_time': end_time - start_decompress_time,
            'CR': getCR(slice_data.shape[0] * slice_data.shape[1], comp_filename) # Not taking RGB into account because image is originally grayscale, and assuming np.uint8
        }
        
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f)


def runJPEGOnKiTS(outputs, quality):
    output_dir = os.path.join(outputs, 'kits')

    for case_num in range(300):
        case_id = "case_{:05d}".format(case_num)

        try:
            case = nib.load(f"kits19-master/{case_id}/imaging.nii.gz")
        except Exception:
            continue

        case_data = case.get_fdata()
        case_data = np.clip(case_data, DEFAULT_HU_MIN, DEFAULT_HU_MAX)

        print(f"[Process KiTS19] {case_id}")
        runJPEGOnCase(case_data, case_id, output_dir, quality, slice_axis=0)
        print(f"[Done] {case_id}: {case_data.shape[0]} slices.")


def runJPEGOnBraTS(outputs, modality, quality):
    output_dir = os.path.join(outputs, 'brats')

    for case_num in range(1667):
        case_id = "BraTS2021_{:05d}".format(case_num)

        try:
            case = nib.load(f"BraTS2021_Training_data/{case_id}/{case_id}_{modality}.nii.gz")
        except Exception:
            continue

        case_data = case.get_fdata()

        print(f"[Process BraTS2021] {case_id}")
        runJPEGOnCase(case_data, case_id, output_dir, quality, slice_axis=2)
        print(f"[Done] {case_id}: {case_data.shape[0]} slices.")


def JPEG():
    '''
    Run the JPEG algorithm, encoding and decoding, on all KiTS data and BRaTS data. Save the binary file, 
    decompressed image, and MSE, PSNR, encoding_time, and decoding_time metrics. Uses a constant QUALITY=50
    '''
    QUALITY = 50
    outputs_dir = 'outputs'

    runJPEGOnBraTS(outputs_dir, 't1ce', QUALITY)
    # runJPEGOnKiTS(outputs_dir, QUALITY)
        

if __name__ == '__main__':
    JPEG()
