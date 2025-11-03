from jpeg_huffman_tables import *
import numpy as np

# def decode_zigzag(zigzag):
#     pass

# def AC_decode(block_code):
#     # If the last characters are the same as the EOB character, remove it
#     if block_code[-AC_HUFFMAN_TABLE[(0,0)]['length']:] == AC_HUFFMAN_TABLE[(0,0)]['code']:
#         block_code = block_code[:-AC_HUFFMAN_TABLE[(0,0)]['length']]


# def block_decode(block_code):
#     zigzag, block_code = AC_decode(block_code)
#     dc_value, previous_dc, block_code = DC_decode(block_code)
#     return previous_dc, zigzag, block_code


# def JPEG_decode(block_code, img_width, img_height, quantization_table):
#     # order already reversed
#     image_data = np.empty((img_height // 8, img_width // 8, 8, 8))
#     for i in range(img_height // 8):
#         for j in range(img_width // 8):
#             previous_dc, zigzag, block_code = block_decode(block_code)
#             block = decode_zigzag(zigzag)
#             image_data[i, j, :, :] = block
            
            
