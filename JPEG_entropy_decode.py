import numpy as np
from JPEG_entropy import ZIGZAG_ORDER
from jpeg_huffman_tables import *


def get_from_huffman(block_code, table):
    """
    Get the first value in a bitstream given the Huffman table to look at
    
    Input:
        block_code - the binary bitstream
        table - the Huffman table to look at
    
    Output:
        key - the first value in the bitstream
        length - the number of bits that the key takes up in the bitstream
    """
    for key, val in table.items():
        if block_code[:val['length']] == val['code']:
            return key, block_code[val['length']:]
    raise Exception


def decode_zigzag(zigzag_sequence):
    """
    Convert a 1D zigzag array into its 8x8 form
    
    Input:
        zigzag_sequence - the 1D zigzag array of 64 values
    
    Output:
        zigzag_arr - the reformatted 2D 8x8 zigzag array
    """
    zigzag_arr = np.empty((8, 8))
    for i in range(64):
        row_num = ZIGZAG_ORDER[i] // 8
        col_num = ZIGZAG_ORDER[i] % 8

        zigzag_arr[row_num, col_num] = zigzag_sequence[i]
    return zigzag_arr


def decode_VLI(bitsize, block_code):
    """
    Get the numerical value of a variable from its bitsize, taking its 2's compliment if necessary
    
    Input:
        bitsize - the bitsize of the number to return
        block_code - the binary bitstream
    
    Output:
        num - the value
        block_code - the remainder of the binary bitstream
    """
    if bitsize == 0:
        return 0, block_code
    
    num = int(block_code[:bitsize], 2)
    if num < 2 ** (bitsize-1): # Was originally negative, then turned positive, so must be turned back
        # two's compliment
        num = num - (1 << bitsize)
        num += + 1
    return num, block_code[bitsize:]


def DC_decode(block_code):
    """
    Get the difference between the current and previous DC values from the bitstream.
    
    Input:
        block_code - the binary bitstream
    
    Output:
        VLI_val - the difference between the DC values
        block_code - the remainder of the binary bitstream
    """
    VLC_val, block_code = get_from_huffman(block_code, DC_HUFFMAN_TABLE) # Get bitsize of VLI_diff
    VLI_val, block_code = decode_VLI(VLC_val, block_code)
    return VLI_val, block_code


def AC_decode(current_dc, block_code):
    """
    Get the AC values from the bitstream.
    
    Input:
        current_dc - the first coefficient in the block
        block_code - the binary bitstream
    
    Output:
        zigzag_sequence - the 1D array of the 64 coefficients in this block
        block_code - the remainder of the binary bitstream
    """
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
    """
    Decode the bitstream to get an array of the values in the original 8x8 array from it.
    
    Input:
        previous_dc - the value of the previous block's DC, or 0 if this is the first block
        block_code - the binary bitstream
    
    Output:
        current_dc - the DC value for this block
        zigzag_sequence - the 1D array of the 64 coefficients in this block
        block_code - the remainder of the binary bitstream
    """
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


def JPEG_decode(input_filename):
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

        # Find DQT (Define Quantization Table) marker
        dqt_index = file_content.find(b'\xFF\xDB')
        
        # Skip marker (2 bytes), length (2 bytes), and precision/table ID (1 byte)
        qt_offset = dqt_index + 5
        
        # Read 128 bytes (64 values * 2 bytes each for 16-bit quantization values)
        qt_data = file_content[qt_offset:qt_offset + 128]
        
        # Parse quantization table from zigzag order to 8x8 array
        Zig = [
            [ 0, 1, 5, 6,14,15,27,28],
            [ 2, 4, 7,13,16,26,29,42],
            [ 3, 8,12,17,25,30,41,43],
            [ 9,11,18,24,31,40,44,53],
            [10,19,23,32,39,45,52,54],
            [20,22,33,38,46,51,55,60],
            [21,34,37,47,50,56,59,61],
            [35,36,48,49,57,58,62,63]
        ]
        
        quantization_table = np.zeros((8, 8), dtype=int)
        for i in range(64):
            # Read 16-bit value (2 bytes)
            qt_value = (qt_data[i*2] << 8) | qt_data[i*2 + 1]
            
            # Find row and column from zigzag position
            for row in range(8):
                for col in range(8):
                    if Zig[row][col] == i:
                        quantization_table[row, col] = qt_value
                        break
        
        #Start of Frame 0 marker
        # indicates the beginning of the frame header which contains image info
        sof0_index = file_content.find(b'\xFF\xC0')
    
        # Skip marker - 2 bytes, length - 2 bytes, and precision - 1 byte
        offset = sof0_index + 5
        
        # Read height
        height = (file_content[offset] << 8) | file_content[offset + 1]
        offset += 2
        
        # Read width
        width = (file_content[offset] << 8) | file_content[offset + 1]
        
        # Calculate number of blocks
        num_blocks_vertical = (height + 7) // 8
        num_blocks_horizontal = (width + 7) // 8

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
    
    return image, quantization_table
