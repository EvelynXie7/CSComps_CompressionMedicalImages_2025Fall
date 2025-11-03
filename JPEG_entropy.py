"""
Author: Justin Vaughn

JPEG Entropy Encoding Implementation
Uses Huffman coding and Run-Length Encoding for compression of quantized DCT coefficients.

Based on Purdue University Digital Image Processing Laboratory:
"Achromatic Baseline JPEG encoding Lab" by Prof. Charles A. Bouman
https://engineering.purdue.edu/~bouman/grad-labs/JPEG-Image-Coding/
put_header and put_tail commented code is directly from the lab and was 
adapted into python.

All encoding functions (DC_encode, AC_encode, VLI_encode, etc.) adapted from 
Purdue Lab Section 3 (Entropy Encoding of Coefficients, pages 7-15) and 
Appendix B (AC Huffman Tables, pages 23-26).




Original work includes Python implementation and integration of the encoding pipeline.
"""

import numpy as np
from quantization import getJPEGQuantizationTable

SOI = 0xFFD8   # Start of Image
EOI = 0xFFD9   # End of Image
DQT = 0xFFDB   # Define Quantization Table
SOF0 = 0xFFC0  # Start of Frame (baseline DCT)
DHT = 0xFFC4   # Define Huffman Table
SOS = 0xFFDA   # Start of Scan


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
    # raster-order index is row*8+ col
    0,  1,  8,  16, 9,  2,  3,  10,
    17, 24, 32, 25, 18, 11, 4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6,  7,  14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63

]

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

# standard quantization table
# need to apply different one for areas not in ROI
# not used yet, person implement quantization can do thius
QUANT = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
]

# =============================================================================
# Helper Functions
# =============================================================================

def BitSize(value)->int:
    """
    Input: value - signed integer
    Output: Integer containing position of the most significant bit in the unsigned value
    """

    # in case value is 0 want length of 1
    if value == 0:
        return 0
    
    value=int(value)
    # default bit_length function on the absolute value
    return abs(value).bit_length()


def VLI_encode(bitsize, value, block_code):
    if bitsize == 0:
        return block_code
    if value >= 0:
        block_code += bin(value)[2:].zfill(bitsize)
    else:
        code = (1 << bitsize) - 1 + value
        block_code += bin(code & ((1 << bitsize) - 1))[2:].zfill(bitsize)
    return block_code


def ZigZag(block):
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
    #convert block coordinates to pixel coordiantes
    zigzag_sequence= []

    for raster_index in ZIGZAG_ORDER:
        row=raster_index // 8
        col = raster_index % 8
        zigzag_sequence.append(block[row,col])

    return zigzag_sequence


# =============================================================================
# DC and AC Encoding Functions
# =============================================================================

def DC_encode(dc_value, prev_value, block_code):
    """
    Encode DC coefficient using Differential Pulse Code Modulation (DPCM).
    
    Input:
        dc_value - DC coefficient value in current JPEG block.
        previous_dc - DC coefficient value in previous JPEG block in raster order.
        block_code - string of current encoded bits
    Output: updated block_code
    """
    # Calculate difference from previous DC value
    diff= dc_value-prev_value
    
    # Get bitsize for the difference
    diff_bitsize=BitSize(diff)

    # Get corresponding VLC code in dc_huffman_table
    VLC_code=DC_HUFFMAN_TABLE[diff_bitsize]['code']
    
    # Add VLC code to block_code
    block_code+=VLC_code
    
    # Add VLI encoding of difference value
    block_code = VLI_encode(diff_bitsize,diff, block_code)
    
    return block_code


def AC_encode(zigzag, block_code):
    """
    Encode AC coefficients using Run-Length Encoding (RLE) + Huffman coding.
    
    Input:
        zigzag - 64 coefficients in zigzag order
        block_code - current encoded bits
        ac_huffman_table - AC Huffman lookup table
    Output: updated block_code
    
    Algorithm pseudocode from Purdue lab (Section 3, pages 13-14):
    """
    
    # AC encode(zigzag, block code) {
    # /* Init variables */
    # int idx = 1 ;
    idx=1
    # int zerocnt = 0 ;
    zerocnt=0
    # int bitsize ;
    bitsize=0
    
    # while(idx < 64) {
    while(idx<64):
        # if(zigzag[idx] == 0) zerocnt ++ ;
        if zigzag[idx]==0:
            zerocnt+=1
        # else {
        else:

            # /* ZRL coding */
            # for( ; zerocnt > 15; zerocnt -= 16)
                # block code ← strcat( block code, acHuffman.code[15][0] );
            while(zerocnt > 15):
                block_code += AC_HUFFMAN_TABLE[(15, 0)]['code']
                zerocnt -= 16
            # bitsize = BitSize(zigzag[idx]) ;
            bitsize= BitSize(zigzag[idx]) 
            # block code ← strcat( block_code, acHuffman.code[zerocnt][bitsize] );
            block_code += AC_HUFFMAN_TABLE[(zerocnt, bitsize)]['code']
            # VLI encode( bitsize, zigzag[idx], block_code) ;
            block_code=VLI_encode(bitsize, zigzag[idx], block_code )
            # zerocnt = 0 ;
            zerocnt=0
        # }
        # idx++ ;
        idx+=1
    # }
    # /* EOB coding */
    # if(zerocnt) block code ← strcat( block code, acHuffman.code[0][0] );
    if(zerocnt):
        block_code += AC_HUFFMAN_TABLE[(0, 0)]['code']

    return block_code


def block_encode(previous_dc, zigzag, block_code):
    """
    calls DC encode followed by AC encode to encode the entire
    block of quantized DCT coefficients.   

    Input:
        previous_dc - DC from previous block (for DPCM)
        zigzag - 64 coefficients in zigzag order
        block_code - current encoded bits

    Output: updated block_code
    
    """
    #encode DC coefficient
    dc_value=zigzag[0]
    block_code=DC_encode(dc_value, previous_dc, block_code)
        
    # use AC encoding function    
    block_code=AC_encode(zigzag, block_code)
    
    return block_code


# =============================================================================
# Bit-to-Byte Conversion Functions
# =============================================================================

def convert_encode(block_code)->tuple:
    """
    Convert string of '0' and '1' characters into actual byte values.
    Implements byte stuffing for JPEG compliance.
    
    Input: 
        block_code - string of '0' and '1' characters
    Output: 
        block_code - character string containing binary characters that have not yet been encoded into the byte code
        array. In general, the returned string will be of length < 8 and will contain the
        trailing bits that would not completely fill a full byte.

        byte_code - the converted output byte sequence produced by mapping the
        characters of block code to the bits of unsigned characters.

        length - The number of bytes in the array byte code.
        include byte stuffing in the final byte sequence
    """
    
    # Process block_code 8 bits at a time
    i=0
    byte_array = bytearray()
    while i + 8 <= len(block_code):
        #Convert binary string to byte
        byte_string = block_code[i:i+8]
        byte_value = int(byte_string, 2)
        #Update byte_array
        byte_array.append(byte_value)
        
        #if byte is 0xFF stuff byte
        if byte_value == 0xFF:
            byte_array.append(0x00)
        
        i += 8
    block_code=block_code[i:]
    length=len(byte_array)
    
    # Return block_code, byte_code, and length as tuple
    return block_code, byte_array, length



def zero_pad(block_code, byte_array, length) -> tuple:
    """
    Pad remaining bytes with 0s to complete final byte.
    
    Input:  
        block_code - A character string containing the remaining bytes after the last JPEG
        block has been encoded. This character string is produced by the Convert encode subroutine.

    Output:  
    byte_array - byte_array with added byte, length
    """
    # Check if there are remaining bytes
    if block_code is None or len(block_code) == 0:
        return byte_array,length
    
    if len(block_code) >= 8:
        raise ValueError("block_code must be less than 8 bits}")
    
    # Calculate padding needed
    padding = 8 - len(block_code)

    #Pad with zeroes
    padded_code = block_code + '0' * padding

    #COnvert to byte and append to byte array
    byte_value = int(padded_code, 2)

    byte_array.append(byte_value)
    
    length+=1

    return byte_array,length

def put_header(width:int, height:int, quant, fileout):
    # void put_header(
    #   int width,        /* number of columns in image */
    #   int height,       /* number of rows in image */
    #   int quant[8][8],  /* 8x8 quantization matrix */
    #   FILE * fileout)
    # {
    #   static unsigned char DC_table[] = 
    #   {0xFF,0xC4,0x00,0x1F,0x00, 
    #    0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00, 
    #    0x00,0x00,0x00,0x00,0x00,0x00,                 /* length table */
    #    0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0A,0x0B /* list part */
    #    } ; 

    DC_table = bytes([
        0xFF, 0xC4, 0x00, 0x1F, 0x00,
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B  # list part
        ])

    #   static unsigned char AC_table[] = 
    #   {0xFF,0xC4,0x00,0xB5,0x10,
    #    0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,
    #    0x05,0x05,0x04,0x04,0x00,0x00,0x01,0x7D,
    #    0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,
    #    0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,
    #    0x22,0x71,0x14,0x32,0x81,0x91,0xA1,0x08,
    #    0x23,0x42,0xB1,0xC1,0x15,0x52,0xD1,0xF0,
    #    0x24,0x33,0x62,0x72,0x82,0x09,0x0A,0x16, 
    #    0x17,0x18,0x19,0x1A,0x25,0x26,0x27,0x28,
    #    0x29,0x2A,0x34,0x35,0x36,0x37,0x38,0x39,
    #    0x3A,0x43,0x44,0x45,0x46,0x47,0x48,0x49,
    #    0x4A,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
    #    0x5A,0x63,0x64,0x65,0x66,0x67,0x68,0x69,
    #    0x6A,0x73,0x74,0x75,0x76,0x77,0x78,0x79,
    #    0x7A,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
    #    0x8A,0x92,0x93,0x94,0x95,0x96,0x97,0x98,
    #    0x99,0x9A,0xA2,0xA3,0xA4,0xA5,0xA6,0xA7,
    #    0xA8,0xA9,0xAA,0xB2,0xB3,0xB4,0xB5,0xB6,
    #    0xB7,0xB8,0xB9,0xBA,0xC2,0xC3,0xC4,0xC5,
    #    0xC6,0xC7,0xC8,0xC9,0xCA,0xD2,0xD3,0xD4,
    #    0xD5,0xD6,0xD7,0xD8,0xD9,0xDA,0xE1,0xE2,
    #    0xE3,0xE4,0xE5,0xE6,0xE7,0xE8,0xE9,0xEA,
    #    0xF1,0xF2,0xF3,0xF4,0xF5,0xF6,0xF7,0xF8,
    #    0xF9,0xFA} ;
    AC_table = bytes([
        0xFF, 0xC4, 0x00, 0xB5, 0x10,
        0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03,
        0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
        0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16,
        0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
        0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
        0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5,
        0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4,
        0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
        0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
        0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8,
        0xF9, 0xFA
    ])

    #   static unsigned char Comment[] =
    #   {0xFF,0xFE,0x00,0x44,0x20,
    #    0x45,0x64,0x75,0x63,0x61,0x74,0x69,0x6f,
    #    0x6e,0x61,0x6c,0x20,0x50,0x75,0x72,0x70,
    #    0x6f,0x73,0x65,0x20,0x4f,0x6e,0x6c,0x79,
    #    0x3b,0x0a,0x20,0x53,0x6f,0x66,0x74,0x77,
    #    0x61,0x72,0x65,0x20,0x63,0x6f,0x70,0x79,
    #    0x72,0x69,0x67,0x68,0x74,0x20,0x50,0x75,
    #    0x72,0x64,0x75,0x65,0x20,0x55,0x6e,0x69,
    #    0x76,0x65,0x72,0x73,0x69,0x74,0x79,0x2e,
    #    0x0a
    #   } ;
    '''
    Comment = bytes([
        0xFF, 0xFE, 0x00, 0x44, 0x20,
        0x45, 0x64, 0x75, 0x63, 0x61, 0x74, 0x69, 0x6f,
        0x6e, 0x61, 0x6c, 0x20, 0x50, 0x75, 0x72, 0x70,
        0x6f, 0x73, 0x65, 0x20, 0x4f, 0x6e, 0x6c, 0x79,
        0x3b, 0x0a, 0x20, 0x53, 0x6f, 0x66, 0x74, 0x77,
        0x61, 0x72, 0x65, 0x20, 0x63, 0x6f, 0x70, 0x79,
        0x72, 0x69, 0x67, 0x68, 0x74, 0x20, 0x50, 0x75,
        0x72, 0x64, 0x75, 0x65, 0x20, 0x55, 0x6e, 0x69,
        0x76, 0x65, 0x72, 0x73, 0x69, 0x74, 0x79, 0x2e,
        0x0a
    ])
    '''
    #    unsigned int qt[64] = {0} ; #quantization table
    qt= [0]*64
    #    int      i,j ;
    i=0
    j=0
    #    unsigned char     p[64]  ;
    p= bytearray(64)

    #    /* Image start header */
    #    p[0] = 0xff ; 
    p[0]=0xff
    #    p[1] = SOI &0xff ;
    p[1]= SOI & 0xff
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])



    #    /* Quant table header */

    #    p[0] = 0xff ; 
    p[0]=0xff
    #    p[1] = DQT &0xff ;
    p[1] = DQT &0xff
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])

    #    /* Lq_h,Lq_l,(Pq,Tq) */
    #    p[0] = 0 ;
    p[0] = 0
    #    p[1] = 0x83 ;
    p[1] = 0x83
    #    p[2] = 0x10 ;
    p[2] = 0x10 
    #    fwrite(p,sizeof(char),3,fileout) ;
    fileout.write(p[:3])

    #    /* Quant table content in zigzag order */
    #    for( i=0 ;i<8; i++ )
    for i in range(8):
    #      for( j=0; j<8; j++ ){
        for j in range(8):
    #        qt[(Zig[i][j])] = quant[i][j] ;
            qt[(Zig[i][j])] = quant[i][j]
    #      }
    #    for( i=0 ; i<64; i++ ){
    for i in range(64):
    #      p[0] = ( qt[i] >> 8 )& 0xff ;
        p[0] = ( qt[i] >> 8 )& 0xff
    #      p[1] = qt[i] & 0xff ; 
        p[1] = qt[i] & 0xff
    #      fwrite(p,sizeof(char),2,fileout) ;
        fileout.write(p[:2])
    #    }

    #    /* Comments */
    #    fwrite(Comment,sizeof(char),sizeof(Comment)/sizeof(Comment[0]),fileout) ; 
    # fileout.write(Comment)

    #    /* Baseline Frame start marker */
    #    p[0] = 0xff ;
    p[0] = 0xff
    #    p[1] = SOF0 & 0xff ;
    p[1] = SOF0 & 0xff 
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])

    #    /*  Lf_h,Lf_l,P,Y_h,Y_l,X_h,X_l */

    #    /* Lf and P */
    #    p[0] = 0 ;
    p[0] = 0
    #    p[1] = 0x0b ;
    p[1] = 0x0b
    #    p[2] = 0x08 ;
    p[2] = 0x08
    #    fwrite(p,sizeof(char),3,fileout) ;
    fileout.write(p[:3])

    #    /* Y */
    #    p[0] = (height >> 8 ) & 0xff ;
    p[0] = (height >> 8 ) & 0xff
    #    p[1] = height & 0xff ;
    p[1] = height & 0xff
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])

    #    /* X */
    #    p[0] = (width >> 8 ) & 0xff ;
    p[0] = (width >> 8 ) & 0xff 
    #    p[1] = width & 0xff ;
    p[1] = width & 0xff 
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])

    #    /* Nf, H1, V1, and Tq1 */
    #    p[0] = 0x1; /* Nf,Ci,(Hi,Vi),Tqi */
    p[0] = 0x1; # Nf,Ci,(Hi,Vi),Tqi
    #    p[1] = 0x1;
    p[1] = 0x1
    #    p[2] = 0x44;
    p[2] = 0x44
    #    p[3] = 0;
    p[3] = 0
    #    fwrite(p,sizeof(char),4,fileout) ;
    fileout.write(p[:4])

    #    /* DHT for luminance DC value category */
    #    fwrite(DC_table,sizeof(char),sizeof(DC_table),fileout) ;
    fileout.write(DC_table)

    #    /* DHT for luminance AC zero-run & value category */
    #    fwrite(AC_table,sizeof(char),sizeof(AC_table),fileout) ;
    fileout.write(AC_table)

    #    /* Start scan segment */
    #    p[0] = 0xff;
    p[0] = 0xff
    #    p[1] = SOS & 0xff;
    p[1] = SOS & 0xff
    #    fwrite(p,sizeof(char),2,fileout) ;
    fileout.write(p[:2])

    #    /* Ls, Ns, Csj */
    #    p[0] = 0x0;
    p[0] = 0x0
    #    p[1] = 0x8;
    p[1] = 0x8
    #    p[2] = 0x01;
    p[2] = 0x01
    #    p[3] = 0x01;
    p[3] = 0x01
    #    fwrite(p,sizeof(char),4,fileout) ;
    fileout.write(p[:4])
    
    #    /* (Tdj, Taj),Ss,Se,(Ah,Al) */
    #    p[0] = 0x0;
    p[0] = 0x0
    #    p[1] = 0x0;
    p[1] = 0x0
    #    p[2] = 0x3F;
    p[2] = 0x3F
    #    p[3] = 0x0;
    p[3] = 0x0
    #    fwrite(p, sizeof(char),4,fileout) ; }
    fileout.write(p[:4])
    
    return

def put_tail(fileout):

    # void put_tail(FILE * fileout)
    # {
    # unsigned char p[2] ;
    p= bytearray(2)

    # p[0] = 0xff;
    p[0] = 0xff
    # p[1] = EOI & 0xff;
    p[1] = EOI & 0xff
    # fwrite(p,sizeof(char),2,fileout) ; 
    # }
    fileout.write(p[:2])
    
    return

# =============================================================================
# Main Encoding Pipeline
# =============================================================================

def JPEG_encode(img, Q):
    """
    Encode the quantized DCT coefficients.
    
    Input: img - list of 8x8 lists (quantized DCT coefficients)
    Output: byte_code - encoded data as a list of bytes
    """

    # block_code = ""
    block_code= ""
    # previous_dc_value = 0
    previous_dc=0

    num_blocks_vertical = img.shape[0]
    num_blocks_horizontal = img.shape[1]
    
    
    # for block in quantized_blocks:
    for i0 in range(num_blocks_vertical):
        for j0 in range(num_blocks_horizontal):
            # Get 8 x 8 block from image
            block = img[i0, j0, :, :]
            
            zigzag_sequence=ZigZag(block)
            
            # call encode function
            block_code = block_encode(previous_dc, zigzag_sequence, block_code)
            
            # Update previous_dc
            previous_dc = zigzag_sequence[0] 


    # Call convert_encode to get byte_code
    # remaining_bits, byte_array, length = convert_encode(block_code)

    # # Call zero_pad
    # byte_array, length = zero_pad(remaining_bits, byte_array, length)
    
    with open("output.bin", 'wb') as fileout:
        #put header
        put_header(num_blocks_horizontal * 8, num_blocks_vertical * 8, getJPEGQuantizationTable(Q), fileout)
        fileout.write(byte_array)

        #put tail
        put_tail(fileout)
    return