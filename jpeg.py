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

# #test function for JPEG I used, wasn't sure how to get yours to run
# def JPEG():
#     QUALITY = 50
    
#     # Load BraTS data
#     nii_path = "/Users/justinvaughn/Downloads/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_flair.nii"
#     nii_img = nib.load(nii_path)
#     original_image_data = nii_img.get_fdata()
    
#     # get slice
#     slice_idx = original_image_data.shape[2] // 2
#     img = original_image_data[:, :, slice_idx]
    
#     # Normalize to 0-255 range
#     max_val=255

#     # Calculate minimum value
#     img_min=img.min()

#     # Calculate maximum value
#     img_max=img.max()

#     # min-max normalization:
#     if img_max == img_min:
#         img_norm= np.zeros_like(img, dtype=np.uint8)
#     else:
#         img_norm = (img-img_min) / (img_max-img_min)

#     img_norm=img_norm * max_val
       
#     compressJPEG(img_norm, QUALITY)

if __name__ == '__main__':
    JPEG()
