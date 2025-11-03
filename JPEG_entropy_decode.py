import numpy as np
from JPEG_entropy import ZIGZAG_ORDER
from jpeg_huffman_tables import *


def get_from_huffman(block_code, table):
    for key, val in table.items():
        if block_code[:val['length']] == val['code']:
            return key, block_code[val['length']:]
    raise Exception

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



def convert_decode(byte_array)-> str:
    """
    Convert byte values into string of '0' and '1' characters.
    Gets rid of byte stuffing.
    
    Input:
        byte_array - array of bytes to decode

    Output:
        block_code - string of '0' and '1' characters

    """
    block_code = ""
    i = 0
    
    while i < len(byte_array):
        byte_value = byte_array[i]
        
        # Convert byte to 8-bit binary string
        block_code += format(byte_value, '08b')
        i += 1
        
        # Check for byte unstuffing
        if byte_value == 0xFF and i < len(byte_array) and byte_array[i] == 0x00:
            # Skip the stuffed 0x00 byte
            i += 1

    return block_code


def JPEG_decode(input_filename, num_blocks_vertical, num_blocks_horizontal):
    """
    Decode JPEG entropy-encoded data back to quantized DCT coefficients.
    Reverses the JPEG_encode function (only the entropy decoding portion).
    
    Input:
        input_filename - path to the binary file containing encoded data
        num_blocks_vertical - number of 8x8 blocks vertically
        num_blocks_horizontal - number of 8x8 blocks horizontally
    
    Output:
        img - 4D numpy array of shape (num_blocks_vertical, num_blocks_horizontal, 8, 8)
              containing decoded quantized DCT coefficients
    """
    
    with open(input_filename, 'rb') as filein:
        file_content = filein.read()
        
        # Find Start of Scan (SOS) marker
        sos_index = file_content.find(b'\xFF\xDA')
        
        # Skip SOS marker and scan header
        scan_header_length = (file_content[sos_index + 2] << 8) | file_content[sos_index + 3]
        data_start = sos_index + 2 + scan_header_length
        
        # Find End of Image (EOI) marker
        eoi_index = file_content.find(b'\xFF\xD9', data_start)
        
        # Extract encoded data between SOS header and EOI
        encoded_data = file_content[data_start:eoi_index]
        
        # Convert bytes to bit string
        block_code = convert_decode(encoded_data)
    
    image = np.zeros((num_blocks_vertical, num_blocks_horizontal, 8, 8), dtype=int)
    current_dc = 0
    
    for i in range(num_blocks_vertical):
        for j in range(num_blocks_horizontal):
            current_dc, zigzag_sequence, block_code = block_decode(current_dc, block_code)
            image[i, j] = decode_zigzag(zigzag_sequence)
    
    return image
