"""
author: Justin Vaughn

Heap based implementation of huffman encoding and decoding.

Help / inspirations:
 __lt__ method directly adapted from GeeksforGeeks Huffman Coding tutorial
Overall algorithm structure and heap approach inspired by GeeksforGeeks (see inner comments for details)
Node class structure follows assignment suggestion

Original work includes added error handling, read_file_make_table(filename), 
adding a Huffman dictionary, encode_text, decode_text, and main function.

GeeksforGeeks Huffman Coding tutorial: https://www.geeksforgeeks.org/huffman-coding-greedy-algo-3/
"""

import heapq


# Class inspired by assignment suggestions
class Node:
    """
    Node class
    instance variables - symbol, frequency, leftChild and rightChild.
    """
    def __init__(self, symbol, frequency, leftChild, rightChild):
        self.symbol = symbol
        self.frequency = frequency
        self.leftChild = leftChild
        self.rightChild = rightChild

    # Directly Adapted from GeeksforGeeks Huffman Coding tutorial
    # Used to compare node objects
    def __lt__(self, other):
        return self.frequency < other.frequency


def read_file_make_table(filename):
    """
    Input: text file
    Output: Dictionary of frequencies for each character
    """
    chars = {}

    try:
        file = open(filename, 'r')

        content=file.read()
        for char in content:
            # increment dictionary value by 1
            if char in chars:
                chars[char] += 1
            # create new dictionary entry and set value to 1
            else:
                chars[char] = 1

        # manually close file
        file.close()
        print(chars)

        return chars

    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


# Algorithm outline inspired by GeeksforGeeks Huffman Coding tutorial
def make_huffman_dict(chars):
    """
    Input: dictionary of character frequencies
    Output: dictionary of Huffman codes
    """

    heap = []
    for symbol, frequency in chars.items():
        # create leaf node for each unique character
        node = Node(symbol, frequency, None, None)
        # build min heap of all leaf nodes
        heapq.heappush(heap, node)

    # repeat tree building until heap only contains one node
    while len(heap) > 1:
        # extract 2 nodes with min frequency from heap
        left_child = heapq.heappop(heap)
        right_child = heapq.heappop(heap)

        # create new internal node with frequency equal to sum of the 2 node frequencies
        merged_freq = left_child.frequency + right_child.frequency
        # first extracted node left child, other extracted right child
        merged_node = Node(None, merged_freq, left_child, right_child)
        # add to min heap
        heapq.heappush(heap, merged_node)

    # pop root node and return
    root_node = heapq.heappop(heap)

    if root_node is None:
        return {}

    codes = {}
    preOrder(root_node, codes, "")
    
    return codes


# Algorithm outline inspired by GeeksforGeeks Huffman Coding tutorial
# chose to make a dictionary of huffman codes instead of directly translating
def preOrder(node, codes, current_code):
    """
    input: node- node to traverse, codes- dictionary of huffman codes, current_code- current huffman code
    output: nothing, updates codes dictionary
    """
    if node is None:
        return

    # Leaf node represents a character
    if node.leftChild is None and node.rightChild is None:
        # assign the symbol key to the current code
        if current_code:
            codes[node.symbol] = current_code

        #handles edge case of one unique character, assigns it a single bit "0"
        else:
            codes[node.symbol] = "0"
        return
    

    # if moving to the left of huffman tree add 0
    preOrder(node.leftChild, codes, current_code + "0")
    # if moving to the right of huffman tree add 1
    preOrder(node.rightChild, codes, current_code + "1")


def encode_text(filename, dictionary):
    """
    Input: File, Huffman codes
    Output: File's text converted into Huffman codes
    """
    encoded_text = ""

    try:
        file = open(filename, 'r')
        content=file.read()

        #corrected while true
        for char in content:
            if char in dictionary:
                encoded_text += dictionary[char]

        file.close()
        return encoded_text

    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def decode_text(filename, dictionary):
    """
    input: encoded_text - text that is huffman encoded, dictionary - symbol keys and corresponding Huffman codes 
    output: text - decoded string
    """
    try:
        file = open(filename, 'r')

        #reverse dictionary so the bits are keys and can search by bits for corresponding symbol
        reversed_dict = {value: key for key, value in dictionary.items()}

        decoded_text = ""
        bits=""
        
        #corrected while true
        content=file.read()
        for bit in content:
            if bit == '':  # End of file
                break
            
            bits+=bit

            if bits in reversed_dict:
                decoded_text+=reversed_dict[bits]
                bits=""

        file.close()
        return decoded_text

    except FileNotFoundError:
        print(f"Error: '{filename}' not found.")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    


def main():
    """
    output- Huffman encoding of a txt file and the decoding of that code
    """
    filename = "Emma.txt"
    
    # Read and create Huffman code dictionary
    char_dict = read_file_make_table(filename)
    codes_map = make_huffman_dict(char_dict)
    
    # Get the encoded text
    encoded_text = encode_text(filename, codes_map)
    
    # Write encoded text to file
    with open("encoded.txt", "w") as file_e:
        file_e.write(encoded_text)
    
    # Decode the text from the encoded file
    decoded_text = decode_text("encoded.txt", codes_map)
    
    # Write decoded text to file
    with open("decoded.txt", "w") as file_d:
        file_d.write(decoded_text)
    
    print(f"Encoded text finished")
    print(f"Decoded text finished")


if __name__ == "__main__":
    main()
