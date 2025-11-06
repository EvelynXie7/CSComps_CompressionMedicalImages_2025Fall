"""
SPIHT_decoder.py 
Author: Rui Shen
Revised by: Evelyn Xie, Claude
Python implementation of SPIHT decoder
"""

import numpy as np

def init_spiht_lists(m, level):
    H, W = m.shape
    bandsize = H // (2 ** level) 
    if bandsize < 2:
        raise ValueError(f"Invalid DWT level={level}: SPIHT requires LL ≥ 2×2.")
    
    LIP, LIS = [], []

    # LIP: all LL coefficients
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # LIS: Exclude coordinates without descedents 
    half = bandsize // 2
    for i in range(bandsize):
        for j in range(bandsize):
            if i < half and j < half:
                continue
            LIS.append([i, j, 0])
    
    return LIP, LIS, []


def func_MySPIHT_Dec(bitstream):
    """
    SPIHT Decoder
    
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
                
                # Get sign bit and initialize to midpoint
                if bitstream[ctr] == 1:
                    m[x, y] = 1.5 * (2**n)
                else:
                    m[x, y] = -1.5 * (2**n)
                ctr += 1
                
                # Move to LSP
                LSP.append([x, y])
            else:  # Not significant
                ctr += 1
                new_LIP.append([x, y])  # Keep in LIP
        
        LIP = new_LIP
        
        # ===== SORTING PASS: LIS =====
        LIS_list = list(LIS)
        idx = 0
        
        while idx < len(LIS_list):
            if ctr >= len(bitstream):
                return m
            
            entry = LIS_list[idx]
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
                            
                            # Get sign bit and initialize to midpoint
                            if bitstream[ctr] == 1:
                                m[ox, oy] = 1.5 * (2**n)
                            else:
                                m[ox, oy] = -1.5 * (2**n)
                            ctr += 1
                            
                            LSP.append([ox, oy])
                        else:  # Child not significant
                            ctr += 1
                            LIP.append([ox, oy])
                    
                    # Check if grandchildren exist
                    if (2*(2*x) < H) and (2*(2*y) < W):
                        LIS_list[idx] = [x, y, 1]  # Convert to Type B
                    else:
                        # No grandchildren, remove from LIS
                        LIS_list.pop(idx)
                        idx -= 1  # Adjust index
                else:  # Not significant
                    ctr += 1
                    # Keep as Type A, no change needed
            
            else:  # Type B
                if bitstream[ctr] == 1:  # Significant
                    ctr += 1
                    
                    # Add four offspring as Type A entries
                    offspring_coords = [
                        (2*x, 2*y),
                        (2*x, 2*y+1),
                        (2*x+1, 2*y),
                        (2*x+1, 2*y+1)
                    ]
                    
                    for ox, oy in offspring_coords:
                        # CRITICAL: Check bounds before adding to LIS
                        if 0 <= ox < H and 0 <= oy < W:
                            LIS_list.append([ox, oy, 0])
                    
                    # Remove this Type B entry 
                    LIS_list.pop(idx)
                    idx -= 1  # Adjust index
                else:  # Not significant
                    ctr += 1
                    # Keep as Type B for next bitplane
            idx += 1
        
        LIS = LIS_list
        
        # ===== REFINEMENT PASS =====
        if n < n_max:
            for i in range(LSP_len_before):    
                if ctr >= len(bitstream):
                    return m
                
                x, y = LSP[i]
                val_sign = np.sign(m[x, y])
                refine_bit = bitstream[ctr]
                ctr += 1
                
                # Refine the value by ±2^(n-1)
                if refine_bit == 1:
                    m[x, y] = m[x, y] + (2**(n-1)) * val_sign
                else:
                    m[x, y] = m[x, y] - (2**(n-1)) * val_sign
        
        # Move to next bitplane
        n -= 1
    
    return m