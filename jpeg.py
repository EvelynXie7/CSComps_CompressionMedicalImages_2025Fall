from preprocessing import *
from quantization import *


def main():
    image_data = loadCT(123, 10)
    # image_data = runDCT(image_data)
    # image_data = quantize(image_data, 'jpeg')

    # image_data = decodeQuantization(image_data, 'jpeg')
    # image_data = decodeDCT(image_data)

    PILImage = toPILImage(image_data)
    saveImage(PILImage)



if __name__ == '__main__':
    main()
