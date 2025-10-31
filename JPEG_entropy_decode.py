import numpy as np

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
    
    img = np.zeros((num_blocks_vertical, num_blocks_horizontal, 8, 8), dtype=int)
    
    bit_position = 0
    previous_dc = 0
    
    # Decode each block
    for i0 in range(num_blocks_vertical):
        for j0 in range(num_blocks_horizontal):
            # Decode one block
            zigzag_sequence, bit_position = block_decode(
                block_code, bit_position, previous_dc
            )
            
            # Update previous DC for next block
            previous_dc = zigzag_sequence[0]
            
            # Convert zigzag sequence back to 8x8 block
            block = zigzag_decode(zigzag_sequence)
            
            # Store in output image
            img[i0, j0, :, :] = block
    
    return img