"""
SPIHT_decoder.py 
Author: Rui Shen
Revised by: Evelyn Xie, Claude
Python implementation of SPIHT decoder following original paper (Said & Pearlman, 1996)
"""

import numpy as np

def init_spiht_lists(m, level):
    """
    Initialize SPIHT lists following original paper (Said & Pearlman, 1996)
    
    In the highest pyramid level (LL band), pixels are grouped in 2×2 blocks.
    In each 2×2 block, one pixel (top-left of the block) has no descendants.
    This is marked with a star (★) in Fig. 2 of the original paper.
    """
    H, W = m.shape
    bandsize = H // (2 ** level) 
    if bandsize < 2:
        raise ValueError(f"Invalid DWT level={level}: SPIHT requires LL ≥ 2×2.")
    
    LIP, LIS = [], []

    # LIP: all LL coefficients
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # LIS: Exclude one pixel per 2×2 group (top-left of each group)
    # Following original SPIHT paper: "in each group, one of them has no descendants"
    # Pattern: exclude pixels where (i % 2 == 0) AND (j % 2 == 0)
    for i in range(bandsize):
        for j in range(bandsize):
            # Check if this pixel is the top-left of a 2×2 group
            if (i % 2 == 0) and (j % 2 == 0):
                continue  # This pixel has no descendants (★ in Fig. 2)
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
        
        # ===== SORTING PASS: LIS (with dynamic growth for Type B offspring) =====
        LIS_list = list(LIS)
        idx = 0
        
        while idx < len(LIS_list):
            if ctr >= len(bitstream):
                return m
            
            entry = LIS_list[idx]
            x, y, typ = entry
            
            # Calculate bandsize for offspring positions
            bandsize = H // (2 ** level)
            
            if typ == 0:  # Type A
                if bitstream[ctr] == 1:  # Significant
                    ctr += 1
                    
                    # Process four offspring based on DWT layout
                    offspring = [
                        (x, y),
                        (x, y + bandsize),
                        (x + bandsize, y),
                        (x + bandsize, y + bandsize)
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
                    next_bandsize = bandsize // 2
                    if next_bandsize >= 1:
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
                    
                    # Add grandchildren (children of the LL child at x,y)
                    next_bandsize = bandsize // 2
                    offspring_coords = [
                        (x, y),
                        (x, y + next_bandsize),
                        (x + next_bandsize, y),
                        (x + next_bandsize, y + next_bandsize)
                    ]
                    
                    for ox, oy in offspring_coords:
                        # CRITICAL: Check bounds before adding to LIS
                        if 0 <= ox < H and 0 <= oy < W:
                            LIS_list.append([ox, oy, 0])
                    
                    # Remove this Type B entry (fully processed)
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