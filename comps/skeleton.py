"""
Author: Justin Vaughn

JPEG Entropy Encoding Implementation
Uses Huffman coding and Run-Length Encoding for compression of quantized DCT coefficients.

Based on Purdue University Digital Image Processing Laboratory:
"Achromatic Baseline JPEG encoding Lab" by Prof. Charles A. Bouman
https://engineering.purdue.edu/~bouman/grad-labs/JPEG-Image-Coding/pdf/lab.pdf

All encoding functions (DC_encode, AC_encode, VLI_encode, etc.) adapted from 
Purdue Lab Section 3 (Entropy Encoding of Coefficients, pages 7-15) and 
Appendix B (AC Huffman Tables, pages 23-26).

Note: This implements only the entropy encoding step. To create complete JPEG files,
additional header/trailer functions (put_header, put_tail) would be needed as 
described in Purdue Lab Section 3.1 and Appendix A.

Original work includes Python implementation and integration of the encoding pipeline.
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

def calculate_bitsize(value)->int:
    """
    Input: value - signed integer
    Output: Integer containing position of the most significant bit in the unsigned value
    """

    # in case value is 0 want length of 1
    
    # can use default bit length function


def VLI_encode(bitsize, value, block_code):
    """
    Encodes an integer using Variable Length Integer (VLI) encoding.
    
    Input: 
        bitsize - number of bits needed (from calculate_bitsize)
        value - integer to encode
        block_code - current string of binary characters
    Output: updated  with VLI bits appended

    VLI encoding rules from Purdue lab Section 3, pages 7-8:
    """
    # If positive value: 
    #   convert to binary and take last bitsize bits
    
    # if negative value:
    #   Get 2's complement representation
    #   convert to binary string

    #add binary string to bitstring
    pass


def ZigZag(img, i0, j0):
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
    
    # Get bitsize for the difference
    
    # Get corresponding VLC code in dc_huffman_table
    
    # Add VLC code to block_code
    
    # Add VLI encoding of difference value
    
    pass


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
    # int zerocnt = 0 ;
    # int bitsize ;
    
    # while( idx < 64 ) {
    # if( zigzag[idx] == 0 ) zerocnt ++ ;
    # else {
        # /* ZRL coding */
        # for( ; zerocnt > 15; zerocnt -= 16)
        # block code ← strcat( block code, acHuffman.code[15][0] );
        # bitsize = BitSize( zigzag[idx] ) ;
        # block code ← strcat( block code, acHuffman.code[zerocnt][bitsize] );
        # VLI encode( bitsize, zigzag[idx], block code ) ;
        # zerocnt = 0 ;
    # }
    # idx ++ ;
    # }
    # /* EOB coding */
    # if(zerocnt) block code ← strcat( block code, acHuffman.code[0][0] );

    pass


def block_encode(previous_dc, zigzag, block_code):
    """
    calls DC encode followed by AC encode to encode the entire
    block of quantized DCT coefficients.   

    Input:
        previous_dc - DC from previous block (for DPCM)
        zigzag_coefficients - 64 coefficients in zigzag order
        block_code - current encoded bits

    Output: updated block_code
    
    """
    # Call DC encoding function
    
    # Call AC encoding function    
    pass


# =============================================================================
# Bit-to-Byte Conversion Functions
# =============================================================================

def convert_encode(block_code, byte_code)->tuple:
    """
    Convert string of '0' and '1' characters into actual byte values.
    Implements byte stuffing for JPEG compliance.
    
    Input: 
        block_code - string of '0' and '1' characters
        byte_code - empty character string
    Output: 
        block_code - character string containing binary characters that have not yet been encoded into the byte code
        array. In general, the returned string will be of length < 8 and will contain the
        trailing bits that would not completely fill a full byte.

        byte_code - the converted output byte sequence produced by mapping the
        characters of block code to the bits of unsigned characters. This output must

        length - The number of bytes in the array byte code.
        include byte stuffing in the final byte sequence
    """
    
    # Process block_code 8 bits at a time
    
    # For each 8-bit chunk:
    #   - Convert binary string to byte
    #   - Update byte_code
    #   - if byte is 0xFF stuff byte
    
    # Return block_code, byte_code, and length as tuple
    
    pass


def zero_pad(block_code)->int:
    """
    Pad remaining bytes with 0s to complete final byte.
    
    Input:  
        block_code - A character string containing the remaining bytes after the last JPEG
        block has been encoded. This string must have length greater than 0 and less than
        8. This character string is produced by the Convert encode subroutine.

    Output:  
    byte_value - converted output byte produced by padding additional zeros to block code.
    """
    # Check if there are remaining bytes
    
    # Pad with 0s to reach 8 bits
    
    # Convert padded string to integer
    
    pass

def put_header(width:int, height:int, quant, file_path):
    
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
    #    unsigned int qt[64] = {0} ;
    #    int      i,j ;
    #    unsigned char     p[64]  ;

    #    /* Image start header */
    #    p[0] = 0xff ; 
    #    p[1] = SOI &0xff ;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /* Quant table header */
    #    p[0] = 0xff ; 
    #    p[1] = DQT &0xff ;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /* Lq_h,Lq_l,(Pq,Tq) */
    #    p[0] = 0 ;
    #    p[1] = 0x83 ;
    #    p[2] = 0x10 ;
    #    fwrite(p,sizeof(char),3,fileout) ;

    #    /* Quant table content in zigzag order */
    #    for( i=0 ;i<8; i++ )
    #      for( j=0; j<8; j++ ){
    #        qt[(Zig[i][j])] = quant[i][j] ;
    #      }
    #    for( i=0 ; i<64; i++ ){
    #      p[0] = ( qt[i] >> 8 )& 0xff ;
    #      p[1] = qt[i] & 0xff ; 
    #      fwrite(p,sizeof(char),2,fileout) ;
    #    }

    #    /* Comments */
    #    fwrite(Comment,sizeof(char),sizeof(Comment)/sizeof(Comment[0]),fileout) ; 

    #    /* Baseline Frame start marker */
    #    p[0] = 0xff ;
    #    p[1] = SOF0 & 0xff ;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /*  Lf_h,Lf_l,P,Y_h,Y_l,X_h,X_l */

    #    /* Lf and P */
    #    p[0] = 0 ;
    #    p[1] = 0x0b ;
    #    p[2] = 0x08 ;
    #    fwrite(p,sizeof(char),3,fileout) ;

    #    /* Y */
    #    p[0] = (height >> 8 ) & 0xff ;
    #    p[1] = height & 0xff ;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /* X */
    #    p[0] = (width >> 8 ) & 0xff ;
    #    p[1] = width & 0xff ;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /* Nf, H1, V1, and Tq1 */
    #    p[0] = 0x1; /* Nf,Ci,(Hi,Vi),Tqi */
    #    p[1] = 0x1;
    #    p[2] = 0x44;
    #    p[3] = 0;
    #    fwrite(p,sizeof(char),4,fileout) ;

    #    /* DHT for luminance DC value category */
    #    fwrite(DC_table,sizeof(char),sizeof(DC_table),fileout) ;

    #    /* DHT for luminance AC zero-run & value category */
    #    fwrite(AC_table,sizeof(char),sizeof(AC_table),fileout) ;

    #    /* Start scan segment */
    #    p[0] = 0xff;
    #    p[1] = SOS & 0xff;
    #    fwrite(p,sizeof(char),2,fileout) ;

    #    /* Ls, Ns, Csj */
    #    p[0] = 0x0;
    #    p[1] = 0x8;
    #    p[2] = 0x01;
    #    p[3] = 0x01;
    #    fwrite(p,sizeof(char),4,fileout) ;

    #    /* (Tdj, Taj),Ss,Se,(Ah,Al) */
    #    p[0] = 0x0;
    #    p[1] = 0x0;
    #    p[2] = 0x3F;
    #    p[3] = 0x0;
    #    fwrite(p, sizeof(char),4,fileout) ;
    # }
    pass

# =============================================================================
# Main Encoding Pipeline
# =============================================================================

def JPEG_encode(quantized_blocks)->str:
    """
    Encode the quantized DCT coefficients.
    
    Input: quantized_blocks - list of 8x8 lists (quantized DCT coefficients)
    Output: byte_code - encoded data as a list of bytes
    

    """
    # Initialize variables
    # block_code = ""
    # previous_dc_value = 0
    
    # for block in quantized_blocks:
    #     call zigzag
    #     call encode function
    #     Update previous_dc
    
    # Call convert_encode to get byte_code
    
    # Call zero_encode and append to byte_code
    
    # Return byte_code

    #put header

    #put tail
    
    pass

def put_tail(file_path):

    # void put_tail(FILE * fileout)
    # {
    # unsigned char p[2] ;

    # p[0] = 0xff;
    # p[1] = EOI & 0xff;
    # fwrite(p,sizeof(char),2,fileout) ; 
    # }
    pass




# =============================================================================
# Testing and Main Function
# =============================================================================

def main():
    """
    Tests the JPEG entropy encoding implementation
    Uses example from Purdue lab Section 3, pages 9-10.
    """
    
    # Test block from Purdue lab
    # Expected encoding: 7F F9 FF 00 3F E7 FD 26
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
    
    
    # Test zigzag reordering
    
    #separate AC and DC encoding
    
    # Test DC encode
    
    # Test AC encode

    # Test full block encoding

    # Test complete JPEG encoding

    print("Finished testing.")

if __name__ == "__main__":
    main()