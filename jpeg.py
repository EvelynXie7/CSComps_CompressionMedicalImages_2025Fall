from loading import *
from dct import *
from quantization import *
from utils import toPILImage, savePILImage
from metrics import *


def main():
    original_image_data = loadCT(123, save=True)
    image_data = runDCT(original_image_data)
    image_data = quantize(image_data, 'jpeg')

    image_data = decodeQuantization(image_data, 'jpeg')
    image_data = decodeDCT(image_data)

    # mse_stat = mse(original_image_data, image_data)
    # psnr_stat = psnr(original_image_data, image_data)

    # PILImage = toPILImage(image_data)
    # savePILImage(PILImage, 'jpeg_ct')

if __name__ == '__main__':
    main()
