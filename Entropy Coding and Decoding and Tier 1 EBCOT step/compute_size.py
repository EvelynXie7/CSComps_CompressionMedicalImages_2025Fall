"""
author: Justin Vaughn

Computes size of the encoded.txt and decoded.txt as well as their compression ratio and space savings. 

Help / inspirations:
None
"""

def get_size(filename):
    """
    Input: text file
    Output: bit size of file
    """
    length=0

    try:
        file = open(filename, 'r')
        is_Huffman_code=True

        content=file.read()
        for char in content:            
            if char != '0' and char != '1':
                is_Huffman_code=False
            
            length+=1
        if is_Huffman_code==True:
            return length
        if is_Huffman_code==False:
            return 8*length
        

        # manually close file
        file.close()

        

    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

def main():

    
    """
    output- size of the encoded.txt and decoded.txt as well as their compression ratio and space savings
    """
    encoded = "encoded.txt"
    decoded= "decoded.txt"
    
    # Read and create Huffman code dictionary
    size_e = get_size(encoded)
    size_d = get_size(decoded)
    
    # Get the encoded text
    # Print the content
    print(f"Size of encoded text in bits: {size_e}")
    print(f"Size of decoded text in bits: {size_d}")
    print(f"Compression ratio: {size_e/size_d}")
    print(f"Space savings: {1-size_e/size_d}")


if __name__ == "__main__":
    main()