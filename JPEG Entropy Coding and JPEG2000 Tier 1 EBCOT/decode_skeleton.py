"""
Author: Justin Vaughn

JPEG Entropy Decoding Implementation
Reverses Huffman coding and Run-Length Encoding to decompress quantized DCT coefficients.

Based on Purdue University Digital Image Processing Laboratory:
"Achromatic Baseline JPEG encoding Lab" by Prof. Charles A. Bouman
https://engineering.purdue.edu/~bouman/grad-labs/JPEG-Image-Coding/pdf/lab.pdf

The Purdue lab provides detailed encoding algorithms (Section 3, pages 7-15). I
reversed those encoding steps to create corresponding decoding functions:
"""

# =============================================================================
# Constants and Lookup Tables
# =============================================================================

#  Import Huffman tables
from jpeg_huffman_tables import DC_HUFFMAN_TABLE, AC_HUFFMAN_TABLE



# Zigzag scan pattern for 8x8 blocks
# Makes coeffciceint ordered in way that zeros grouped together
ZIGZAG_ORDER = [
    # Need 64 indices of the zigzag pattern
    # see Figure 4(b) from Purdue lab, page 6 for pattern
    # list of 64 numbers representing the raster-order index
    # raster-order indedx is row*8+ col
]


# =============================================================================
# Helper Functions
# =============================================================================



def VLI_decode(bitsize, VLI_block_code):
    """
    Decodes an integer from Variable Length Integer (VLI) encoding.
    
    Input: 
        bitsize - number of bits in encoded value
        VLI_block_code - binary string to decode
    Output: value - decoded integer

    VLI encoding rules from Purdue lab Section 3, pages 7-8:
    """
    #concert to integer
    
    # If positive value: 
    #   return value
    
    # if negative value:
    #   reverse 2's complement
    #   return value

    #add binary string to bitstring
    pass


def ZigZag_Deorder(img, i0, j0):
    """
    Gets 8x8 block from larger image and reorders into zigzag sequence.
    
    Input: 
        img - The full set of DCT coefficients
        i0 - The row number of the JPEG block
        j0 - The column number of the JPEG block.
        
    Output: - A 1-D list of 64 integers containing the DCT coefficients for the JPEG
            block starting at position (i0, j0) in zig zag order.
    
    Implementation from Purdue Lab Section 3.1, page 12.
    Uses ZIGZAG_ORDER lookup table from Figure 4(b), page 6.
    """
    # get 8x8 block at position (i0,j0)
    # Reorder that block into zigzag squence
    # return result
    pass


# =============================================================================
# DC and AC Encoding Functions
# =============================================================================

def DC_decode(block_code, prev_dc):
    """
    Encode DC coefficient using Differential Pulse Code Modulation (DPCM).
    
    Input:
    block_code - binary string of characters to decode
    previous_dc - DC coefficient value in previous JPEG block in raster order.

    Output: 
    dc_value - DC coefficient value in current JPEG block.
    block_code - string of current encoded bits
    remaining_block_code - block_code with decoded bits removed
    """
    # Read bits from block_code until finds a matching Huffman code
    # That code tells you the bitsize
    # Read next 'bitsize' bits and VLI decode them to get difference
    # Calculate DC = prev_dc + difference
    
        
    pass


def AC_decode(block_code):
    """
    Encode AC coefficients using Run-Length Encoding (RLE) + Huffman coding.
    
    Input:
        block_code - encoded bits
    Output: 
        ac_coefficients - list of 63 AC coefficients
        remianing_block_code - block_code wtih decoded bits_removed
    
    Algorithm pseudocode from Purdue lab (Section 3, pages 13-14):
    """
    
    # while(coefficeints<63 or !EOB)
        # read from block_code until find Huffman code

        # get (run_length, bitsize)

        # if (0,0) -> EOB, fill rest with zeroes

        # if (15,0) -> ZRL, add 16 zeroes
        
        # read bitsize bits, VLE decode to get coefficient value

        # insert run_length zeroes before the coefficeint



    pass


def block_decode(previous_dc, zigzag, block_code):
    """
    calls DC encode followed by AC encode to encode the entire
    block of quantized DCT coefficients.   

    Input:
        previous_dc - DC from previous block (for DPCM)
        zigzag_coefficients - 64 coefficients in zigzag order
        block_code - current encoded bits

    Output: updated block_code
    
    """
    # Call reverse_DC encoding function
    
    # Call reverse_AC encoding function    
    pass


# =============================================================================
# Bit-to-Byte Conversion Functions
# =============================================================================

def convert_decode(block_code, byte_code)->str:
    """
    Convert string of '0' and '1' characters into actual byte values.
    Implements byte stuffing for JPEG compliance.
    
    Input: 
        block_code - character string containing binary characters that have not yet been encoded into the byte code
        array. In general, the returned string will be of length < 8 and will contain the
        trailing bits that would not completely fill a full byte.

        byte_code - the converted output byte sequence produced by mapping the
        characters of block code to the bits of unsigned characters. This output must

        length - The number of bytes in the array byte code.
        include byte stuffing in the final byte sequence

    Output:
        block_code - string of '0' and '1' characters

    """
    
    # Process block_code 1 byte at a time
    
    # For each byte:
    #   - Convert byte to binary sequence
    #   - Update block_code
    #   - if byte is 0xFF remove stuff byte    
    # 
    # Return block code
    
    pass


# =============================================================================
# Main Encoding Pipeline
# =============================================================================

def JPEG_decode(byte_code, num_blocks, width_blocks, height_blocks)->str:
    """
    Decode back into quantized DCT coefficients.
    
    Input: 
        byte_code - list of compressed bytes
        num_blocks - total number of 8x8 blocks in image
        width_blocks - number of blocks horizontally
        height_blocks - number of blocks vertically
    Output: 
        quantized_blocks - list of 8x8 lists (quantized DCT coefficients)
    """
    # Convert bytes to bit string and remvoe byte stuffing
    # Decode each with DC + AC decoder
    # COnvert zigzag order back to 8x8 blocks
    
    pass


# =============================================================================
# Testing and Main Function
# =============================================================================

def main():
    """
    Tests the JPEG entropy encoding implementation
    Uses example from Purdue lab Section 3, pages 9-10.
    """
    
    # Test encoded block from Purdue lab
    
    # Block to decode: 7F F9 FF 00 3F E7 FD 26
    
    # Test zigzag reordering
    
    # separate AC and DC encoding
    
    # Test DC encode
    
    # Test AC encode

    # Test full block encoding

    # Test complete JPEG decoding

    print("Finished testing.")

if __name__ == "__main__":
    main()