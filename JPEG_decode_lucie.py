from jpeg_huffman_tables import *
from JPEG_entropy import ZIGZAG_ORDER
import numpy as np

def get_from_huffman(block_code, table):
    for key, val in table.items():
        if block_code[:val['length']] == val['code']:
            return key, block_code[val['length']:]
    print(block_code)
    exit()

def decode_zigzag(zigzag_sequence):
    zigzag_arr = np.empty((8, 8))
    for i in range(64):
        row_num = ZIGZAG_ORDER[i] // 8
        col_num = ZIGZAG_ORDER[i] % 8

        zigzag_arr[row_num, col_num] = zigzag_sequence[i]
    return zigzag_arr


def decode_VLI(bitsize, block_code):
    # Get a value from a bitsize
    if bitsize == 0:
        return 0, block_code
    
    num = int(block_code[:bitsize], 2)
    if num < 2 ** (bitsize-1): # Was originally negative, then turned positive, so must be turned back
        # two's compliment
        num = num - (1 << bitsize)
        num += + 1
    return num, block_code[bitsize:]


def DC_decode(block_code):
    VLC_val, block_code = get_from_huffman(block_code, DC_HUFFMAN_TABLE) # Get bitsize of VLI_diff
    VLI_val, block_code = decode_VLI(VLC_val, block_code)
    return VLI_val, block_code


def AC_decode(current_dc, block_code):
    zigzag_sequence = [current_dc]

    while len(zigzag_sequence) < 64:
        value, block_code = get_from_huffman(block_code, AC_HUFFMAN_TABLE)

        if value == (0, 0): # EOB marker
            return zigzag_sequence + [0 for _ in range(64 - len(zigzag_sequence))], block_code

        zeroes_preceeding_val, val_bitsize = value
        value, block_code = decode_VLI(val_bitsize, block_code)
        zigzag_sequence += [0 for _ in range(zeroes_preceeding_val)] + [value]

    return zigzag_sequence, block_code


def block_decode(previous_dc, block_code):
    VLI_val, block_code = DC_decode(block_code)
    current_dc = previous_dc + VLI_val
    zigzag_sequence, block_code = AC_decode(current_dc, block_code)
    
    return current_dc, zigzag_sequence, block_code


def JPEG_decode(block_code, img_width, img_height):
    image = np.empty((img_width // 8, img_height // 8, 8, 8))
    current_dc = 0
    for i in range(img_width // 8):
        for j in range(img_height // 8):
            current_dc, zigzag_sequence, block_code = block_decode(current_dc, block_code)
            image[i, j] = decode_zigzag(zigzag_sequence)
    return image

def main():
    pass

if __name__ == '__main__':
    main()