"""
Author: [Your Name]

JPEG Entropy Encoding Implementation
Uses Huffman coding and Run-Length Encoding for compression of quantized DCT coefficients.

Based on ITU-T T.81 JPEG Standard and Purdue University JPEG Lab exercises.

Help / Inspirations:
- Purdue University Digital Image Processing Lab: JPEG Encoding
  https://engineering.purdue.edu/~bouman/grad-labs/JPEG-Image-Coding/pdf/lab.pdf
- ITU-T T.81 JPEG Standard (Annex F and K for entropy coding examples)
- JPEG Huffman table structure follows standard baseline tables

Original work includes implementing VLI encoding, zigzag reordering, DC/AC coefficient
encoding, byte stuffing, and integration with quantized DCT blocks.
"""

# TODO:
#  Import your Huffman tables
# from jpeg_huffman_tables import DC_HUFFMAN_TABLE, AC_HUFFMAN_TABLE
import jpeg_huffman_tables


# =============================================================================
# Constants and Lookup Tables
# =============================================================================

# Zigzag scan pattern for 8x8 blocks
# Based on Figure 4(b) from Purdue lab, page 6
# Purpose: reorder coefficients so zeros are grouped together
ZIGZAG_ORDER = [
    # TODO: Fill in the 64 indices for zigzag pattern
    # Hint: Pattern goes (0,0) -> (0,1) -> (1,0) -> (2,0) -> (1,1) -> (0,2) -> ...
    # Should be a list of 64 numbers representing the raster-order index
]


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_bitsize(value):
    """
    Calculate number of bits needed to represent a value's magnitude.
    
    Input: value - signed integer (DC difference or AC coefficient)
    Output: number of bits needed (0-11 for DC, 0-10 for AC)
    
    Algorithm from Purdue lab Section 3, page 7:
    - Find position of most significant bit in absolute value
    - Special case: 0 requires 0 bits
    - Example: 7 needs 3 bits, -7 also needs 3 bits
    """
    # TODO: Implement bitsize calculation
    if value==0:
        return 1
    # Can use bit_length() method or manual bit counting
    return value.bit_length()


def variable_length_integer_encode(bitsize, value, bit_string):
    """
    Encode a signed integer using Variable Length Integer (VLI) encoding.
    
    Input: 
        bitsize - number of bits needed (from calculate_bitsize)
        value - the actual signed integer to encode
        bit_string - current string of '0' and '1' characters
    Output: updated bit_string with VLI bits appended
    
    VLI encoding rules (Purdue lab Section 3, pages 7-8):
    - Positive values: take the 'bitsize' least significant bits
    - Negative values: 
        Step 1: Calculate (value - 1)
        Step 2: Take 2's complement (use 16-bit arithmetic)
        Step 3: Take 'bitsize' least significant bits
    
    Example from Purdue:
        value = 3, bitsize = 2 -> append "11"
        value = -3, bitsize = 2 -> append "00"
            (because 2's comp of -4 is ...11111100, LSB 2 bits = "00")
    """
    # TODO: Implement VLI encoding
    # Handle special case where bitsize is 0
    if bitsize==0:
        raise ValueError("Cannot encode value with 0 bits")
    
    # For positive values:
    #   - Convert to binary and take last 'bitsize' bits
    result = value & ((1 << bitsize) - 1)+1
    binary=bin(result)
    
    # For negative values:
    if value<0:
    #   - Calculate value - 1
        result=value-1
    #   - Get 2's complement representation
        twos_comp = (result & ((1 << bitsize) - 1))
    #   - Convert to binary string
        binary=bin(twos_comp)
    bit_string+=binary[2:]
    pass


def reorder_zigzag(block_2d):
    """
    Reorder 8x8 block from 2D array into 1D zigzag sequence.
    
    Input: block_2d - 8x8 array of quantized DCT coefficients
    Output: list of 64 coefficients in zigzag order
    
    Purpose (Purdue lab Section 2.2, page 6):
    - Groups low-frequency coefficients (likely non-zero) at beginning
    - Groups high-frequency coefficients (likely zero) at end
    - Improves run-length encoding efficiency
    
    Implementation approach:
    - Flatten 2D block to 1D in raster order
    - Use ZIGZAG_ORDER lookup table to reorder elements
    """
    # TODO: Convert 8x8 block to zigzag-ordered 1D array
    # Step 1: Flatten block from 2D to 1D (row-major order)
    # Step 2: Reorder using ZIGZAG_ORDER indices
    pass


# =============================================================================
# DC and AC Encoding Functions
# =============================================================================

def encode_dc_coefficient(dc_value, previous_dc, bit_string, dc_huffman_table):
    """
    Encode DC coefficient using Differential Pulse Code Modulation (DPCM).
    
    Input:
        dc_value - current block's DC coefficient
        previous_dc - previous block's DC coefficient (0 for first block)
        bit_string - current encoded bits
        dc_huffman_table - DC Huffman lookup table
    Output: updated bit_string
    
    DPCM encoding steps (Purdue lab Section 2.2, page 6):
    1. Calculate difference: DIFF = dc_value - previous_dc
    2. Determine bitsize needed for DIFF
    3. Look up Huffman code (VLC) for that bitsize
    4. Append VLC to bit_string
    5. Append VLI encoding of DIFF
    
    Why use DPCM?
    - Adjacent 8x8 blocks have similar average values
    - Encoding differences produces smaller numbers
    - Better compression than encoding absolute values
    """
    # TODO: Implement DC encoding
    # Calculate difference from previous DC value
    
    # Get bitsize for the difference
    
    # Look up VLC code in dc_huffman_table
    # The table maps bitsize -> {'code': '...', 'length': n}
    
    # Append VLC code to bit_string
    
    # Append VLI encoding of difference value
    
    pass


def encode_ac_coefficients(zigzag_array, bit_string, ac_huffman_table):
    """
    Encode AC coefficients using Run-Length Encoding (RLE) + Huffman coding.
    
    Input:
        zigzag_array - 64 coefficients in zigzag order (index 0 is DC, skip it)
        bit_string - current encoded bits
        ac_huffman_table - AC Huffman lookup table
    Output: updated bit_string
    
    Algorithm pseudocode from Purdue lab (Section 3, pages 13-14):
    
    Start at index 1 (skip DC at index 0)
    zero_count = 0
    
    For each coefficient from index 1 to 63:
        If coefficient is zero:
            Increment zero_count
        Else:
            # Found non-zero coefficient
            
            # Handle runs longer than 15 zeros
            While zero_count > 15:
                Append ZRL symbol (15, 0) code
                Subtract 16 from zero_count
            
            # Encode the (run_length, bitsize) pair
            Get bitsize of current coefficient
            Look up Huffman code for (zero_count, bitsize) in ac_huffman_table
            Append that code (VLC)
            Append VLI encoding of coefficient value
            
            Reset zero_count to 0
    
    # After loop, if there are trailing zeros
    If zero_count > 0:
        Append EOB symbol (0, 0) code
    
    Special symbols:
    - (0, 0) = EOB (End of Block) - signals rest of block is zeros
    - (15, 0) = ZRL (Zero Run Length) - represents exactly 16 zeros
    """
    # TODO: Implement AC encoding following the pseudocode above
    # Start at index 1 to skip DC coefficient
    
    # Track consecutive zeros
    
    # Loop through all 63 AC coefficients
    
    # For each zero: increment counter
    
    # For each non-zero:
    #   - Handle runs > 15 with ZRL
    #   - Get bitsize
    #   - Look up (run_length, bitsize) in ac_huffman_table
    #   - Append VLC
    #   - Append VLI
    #   - Reset zero counter
    
    # After loop: append EOB if needed
    
    pass


def encode_block(previous_dc, zigzag_coefficients, bit_string, 
                 dc_huffman_table, ac_huffman_table):
    """
    Encode complete 8x8 block (DC coefficient + 63 AC coefficients).
    
    Input:
        previous_dc - DC from previous block (for DPCM)
        zigzag_coefficients - 64 coefficients in zigzag order
        bit_string - current encoded bits
        dc_huffman_table - DC Huffman table
        ac_huffman_table - AC Huffman table
    Output: updated bit_string
    
    Simple wrapper function (Purdue lab Section 3.1, page 14):
    - First encode DC coefficient
    - Then encode AC coefficients
    """
    # TODO: Call DC encoding function
    # Extract DC value (first element of zigzag_coefficients)
    
    # TODO: Call AC encoding function
    # Pass the zigzag array to encode remaining 63 coefficients
    
    pass


# =============================================================================
# Bit-to-Byte Conversion Functions
# =============================================================================

def convert_bits_to_bytes(bit_string):
    """
    Convert string of '0' and '1' characters into actual byte values.
    Implements byte stuffing for JPEG compliance.
    
    Input: bit_string - string of '0' and '1' characters
    Output: 
        bytes_list - list of byte values (integers 0-255)
        remaining_bits - leftover bits that don't form complete byte
    
    Byte stuffing rule (Purdue lab Section 3, page 10):
    - When a byte equals 0xFF, insert 0x00 immediately after
    - Why? 0xFF is reserved for JPEG markers (SOI, EOI, DHT, etc.)
    - This prevents decoder from mistaking data for markers
    
    Algorithm:
    1. Process 8 bits at a time
    2. Convert each 8-bit string to integer (0-255)
    3. Add byte to output list
    4. If byte is 0xFF, also add 0x00 (byte stuffing)
    5. Return bytes and any remaining bits (< 8)
    """
    # TODO: Implement bit-to-byte conversion with byte stuffing
    # Create empty list for bytes
    
    # Process string 8 bits at a time
    
    # For each 8-bit chunk:
    #   - Convert binary string to integer
    #   - Append to bytes list
    #   - Check if byte is 0xFF and stuff if needed
    
    # Return bytes list and remaining bits
    
    pass


def pad_final_bits(remaining_bits):
    """
    Pad remaining bits with 1s to complete final byte.
    
    Input: remaining_bits - string of '0' and '1' (length < 8)
    Output: final byte value (0-255), or None if no remaining bits
    
    JPEG padding convention (Purdue lab Section 3, page 10):
    - Pad with '1' bits, NOT '0' bits
    - Why? Huffman codes of all 1s are forbidden in JPEG
    - This makes padding distinguishable from actual data
    - Decoder can recognize and ignore the padding
    
    Example:
        remaining_bits = "101" 
        -> pad to "10111111" 
        -> return integer 191
    """
    # TODO: Implement padding
    # Check if there are any remaining bits
    
    # Pad with '1's to reach 8 bits
    # Hint: use string padding methods
    
    # Convert padded string to integer
    
    pass


# =============================================================================
# Main Encoding Pipeline
# =============================================================================

def encode_jpeg_image(quantized_blocks, dc_huffman_table, ac_huffman_table):
    """
    Encode entire image worth of quantized DCT blocks.
    
    Input:
        quantized_blocks - list of 8x8 arrays (quantized DCT coefficients)
        dc_huffman_table - DC Huffman coding table
        ac_huffman_table - AC Huffman coding table
    Output: list of bytes representing compressed entropy-coded data
    
    Processing pipeline:
    1. Initialize empty bit string and previous DC value (0)
    2. For each 8x8 block:
        a. Convert block to zigzag order
        b. Encode block (DC + AC) into bit string
        c. Update previous DC for next block
    3. Convert accumulated bits to bytes
    4. Pad final byte if necessary
    5. Return byte array
    """
    # TODO: Initialize variables
    # accumulated_bits = ""
    # previous_dc_value = 0
    
    # TODO: Process each block
    # for block in quantized_blocks:
    #     Convert block to zigzag
    #     Encode the block
    #     Update previous_dc_value
    
    # TODO: Convert all bits to bytes
    
    # TODO: Handle final padding
    
    # TODO: Return complete byte array
    
    pass


# =============================================================================
# Testing and Main Function
# =============================================================================

def main():
    """
    Test the JPEG entropy encoding implementation.
    Uses example from Purdue lab Section 3, pages 9-10.
    """
    
    # Test block from Purdue lab
    # Expected to encode as: 7F F9 FF 00 3F E7 FD 26 (hex bytes)
    test_block = [
        [ 3,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  9,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0,  0,  0]
    ]
    
    print("Testing JPEG Entropy Encoding...")
    print("=" * 50)
    
    # TODO: Import or define Huffman tables
    # dc_table = DC_HUFFMAN_TABLE
    # ac_table = AC_HUFFMAN_TABLE
    
    # TODO: Test zigzag reordering
    # zigzag = reorder_zigzag(test_block)
    # print(f"Zigzag coefficients: {zigzag}")
    # Expected: [3, 0, 0, 0, ...]
    
    # TODO: Test block encoding
    # bits = ""
    # bits = encode_block(0, zigzag, bits, dc_table, ac_table)
    # print(f"\nEncoded bit string length: {len(bits)}")
    # print(f"First 50 bits: {bits[:50]}")
    
    # TODO: Test byte conversion
    # byte_array, remaining = convert_bits_to_bytes(bits)
    # print(f"\nBytes (hex): {' '.join(f'{b:02X}' for b in byte_array)}")
    # Expected: 7F F9 FF 00 3F E7 FD 26
    
    # TODO: Test padding if needed
    # if remaining:
    #     final_byte = pad_final_bits(remaining)
    #     if final_byte is not None:
    #         byte_array.append(final_byte)
    #         print(f"Final byte (after padding): {final_byte:02X}")
    
    print("\nTesting complete!")


if __name__ == "__main__":
    main()