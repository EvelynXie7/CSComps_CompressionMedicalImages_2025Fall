from preprocessing import *
from dct import *
from quantization import *
from utils import toPILImage, savePILImage


def main():
    image_data = loadCT(123, save=True)
    image_data = runDCT(image_data)
    image_data = quantize(image_data, 'jpeg')

    image_data = decodeQuantization(image_data, 'jpeg')
    image_data = decodeDCT(image_data)
    PILImage = toPILImage(image_data)
    
    savePILImage(PILImage, 'jpeg_ct')

if __name__ == '__main__':
    main()
