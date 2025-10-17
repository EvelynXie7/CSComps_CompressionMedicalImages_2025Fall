from preprocessing import *
from dwt import *
from quantization import *
from utils import *


def main():
    image_data = loadCT(123, save=True)
    image_data = runDWT(image_data)

    image_data = decodeDWT(image_data)
    PILImage = toPILImage(image_data)
    
    savePILImage(PILImage, 'decompressed_ct_spiht')

if __name__ == '__main__':
    main()
