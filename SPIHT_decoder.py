"""
SPIHT_decoder.py - Fixed Version
Python implementation of SPIHT decoder
"""

import numpy as np

def init_spiht_lists(m, level):
    H, W = m.shape
    bandsize = H // (2 ** level) 
    if bandsize <= 1:
        raise ValueError(f"Invalid DWT level={level}: SPIHT requires LL ≥ 2×2.")
    LIP, LIS = [], []

    # LIP: all LL coefficients
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # LIS: remove 1/4 top-left region
    if bandsize == 2:
        for i in range(bandsize):
            for j in range(bandsize):
                if i == 0 and j == 0:
                    continue
                LIS.append([i, j, 0])
    else:
        block_size = bandsize // 2
        skip_size = bandsize // 4
        for i in range(bandsize):
            for j in range(bandsize):
                if (i % block_size) < skip_size and (j % block_size) < skip_size:
                    continue
                LIS.append([i, j, 0])
    return LIP, LIS, []


def func_MySPIHT_Dec(bitstream):
    """
    SPIHT Decoder (Fixed Version)
    
    Parameters:
    bitstream : numpy array
        Bit stream containing encoded image data
    
    Returns:
    m : numpy array
        Reconstructed image in wavelet domain
    """
    H = int(bitstream[0])
    W = int(bitstream[1])
    n_max = int(bitstream[2])
    level = int(bitstream[3])
    ctr = 4
    m = np.zeros((H, W)) 
    LIP, LIS, LSP = init_spiht_lists(m, level)
    
    n = n_max
    
    while ctr < len(bitstream) and n >= 0:
        LSP_len_before = len(LSP)
        
        # ===== SORTING PASS: LIP =====
        new_LIP = []
        for entry in LIP:
            if ctr >= len(bitstream):
                return m
            
            x, y = entry
            if bitstream[ctr] == 1:  # Significant
                ctr += 1
                if ctr >= len(bitstream):
                    return m
                
                # Get sign bit
                if bitstream[ctr] > 0:
                    m[x, y] = 2**n + 2**(n-1) 
                else:
                    m[x, y] = -2**n - 2**(n-1)
                ctr += 1
                
                # Move to LSP
                LSP.append([x, y])
            else:  # Not significant
                ctr += 1
                new_LIP.append([x, y])  # Keep in LIP
        
        LIP = new_LIP
        
        # ===== SORTING PASS: LIS =====
        new_LIS = []
        for entry in LIS:
            if ctr >= len(bitstream):
                return m
            
            x, y, typ = entry
            
            if typ == 0:  # Type A
                if bitstream[ctr] == 1:  # Significant
                    ctr += 1
                    
                    # Process four offspring
                    offspring = [
                        (2*x, 2*y),
                        (2*x, 2*y+1),
                        (2*x+1, 2*y),
                        (2*x+1, 2*y+1)
                    ]
                    
                    for ox, oy in offspring:
                        if ctr >= len(bitstream):
                            return m
                        if not (0 <= ox < H and 0 <= oy < W):
                            continue
                        
                        if bitstream[ctr] == 1:  # Child is significant
                            ctr += 1
                            if ctr >= len(bitstream):
                                return m
                            
                            # Get sign bit
                            if bitstream[ctr] == 1:
                                m[ox, oy] = 2**n + 2**(n-1)
                            else:
                                m[ox, oy] = -2**n - 2**(n-1)
                            ctr += 1
                            
                            LSP.append([ox, oy])
                        else:  # Child not significant
                            ctr += 1
                            LIP.append([ox, oy])
                    
                    # Check if grandchildren exist
                    if (2*(2*x) < H) and (2*(2*y) < W):
                        new_LIS.append([x, y, 1])  # Convert to Type B
                else:  # Not significant
                    ctr += 1
                    new_LIS.append([x, y, 0])  # Keep as Type A
            
            else:  # Type B
                if bitstream[ctr] == 1:  # Significant
                    ctr += 1
                    
                    # Add four offspring as Type A entries
                    offspring = [
                        [2*x, 2*y, 0],
                        [2*x, 2*y+1, 0],
                        [2*x+1, 2*y, 0],
                        [2*x+1, 2*y+1, 0]
                    ]
                    # Add offspring for NEXT bitplane processing
                    new_LIS.extend(offspring)
                else:  # Not significant
                    ctr += 1
                    new_LIS.append([x, y, 1])  # Keep as Type B
        
        LIS = new_LIS
        
        # ===== REFINEMENT PASS =====
        if n < n_max:
            for i in range(LSP_len_before):    
                if ctr >= len(bitstream):
                    return m
                
                x, y = LSP[i]
                val_sign = np.sign(m[x, y])
                refine_bit = bitstream[ctr]
                ctr += 1
                
                # Refine the value
                if refine_bit == 1:
                    m[x, y] = m[x, y] + (2**(n-1)) * val_sign
                else:
                    m[x, y] = m[x, y] - (2**(n-1)) * val_sign
        
        # Move to next bitplane
        n -= 1
    
    return m


# Test code
if __name__ == "__main__":
    bitstream = np.array([
        8, 8, 1, 2,
        1, 1, 0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        0, 0, 0,
        1
    ], dtype=np.int32)

    result = func_MySPIHT_Dec(bitstream)
    print("Decoded Image:")
    print(result)
    
    expected_m = np.array([
        [3.5, 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 0.],
    ], dtype=float)
    print("\nExpected:")
    print(expected_m)
    print(f"\nMatch: {np.allclose(result, expected_m)}")