from loading import *
from dwt import *
from quantization import *
from utils import *


def main():
    # Load image
    image_data = loadCT(123, save=True)

    # Run encoding algorithm
    image_data = runDWT(image_data)

    # Run decoding algorithm
    image_data = decodeDWT(image_data)

    # Save new image
    PILImage = toPILImage(image_data)
    savePILImage(PILImage, 'ct_spiht')

if __name__ == '__main__':
    main()
