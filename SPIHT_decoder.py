"""
SPIHT Decoder 
--------------------------------------------
Author: Rui Shen
Revised by: Evelyn Xie, Claude
- Matches updated encoder logic with correct spatial-orientation tree
- Uses correct offspring structure for deepest LL vs HL/LH/HH bands
- Supports multi-level DWT
- Maintains Type A/B LIS structure and star pattern init

"""

import numpy as np

from SPIHT_encoder import *

#  Decoder 


def func_MySPIHT_Dec(bitstream):
    """
    SPIHT Decoder matching the updated encoder.
    Reconstructs wavelet-domain coefficients from bitstream.
    """
    H = int(bitstream[0])
    W = int(bitstream[1])
    n_max = int(bitstream[2])
    level = int(bitstream[3])
    ctr = 4
    m = np.zeros((H, W))
    bandsize = H // (2 ** level)
    LIP, LIS, LSP = init_spiht_lists(m, level)

    n = n_max

    while ctr < len(bitstream) and n >= 0:
        LSP_len_before = len(LSP)
        new_LIP = []

        # ===== SORTING PASS: LIP =====
        for (x, y) in LIP:
            if ctr >= len(bitstream):
                return m
            if bitstream[ctr] == 1:  # significant
                ctr += 1
                if ctr >= len(bitstream): return m
                sign = bitstream[ctr]; ctr += 1
                m[x, y] = (1.5 if sign == 1 else -1.5) * (2**n)
                LSP.append([x, y])
            else:
                ctr += 1
                new_LIP.append([x, y])
        LIP = new_LIP

        # ===== SORTING PASS: LIS =====
        LIS_list = list(LIS)
        idx = 0
        while idx < len(LIS_list):
            if ctr >= len(bitstream):
                return m
            x, y, typ = LIS_list[idx]

            if typ == 0:  # Type A: process direct offspring
                if bitstream[ctr] == 1:  # offspring are significant
                    ctr += 1
                    offspring = get_offspring(x, y, level, bandsize, H, W)
                    
                    for ox, oy in offspring:
                        if ctr >= len(bitstream): return m
                        if not (0 <= ox < H and 0 <= oy < W): continue
                        
                        if bitstream[ctr] == 1:  # this offspring is significant
                            ctr += 1
                            if ctr >= len(bitstream): return m
                            sign = bitstream[ctr]; ctr += 1
                            m[ox, oy] = (1.5 if sign == 1 else -1.5) * (2**n)
                            LSP.append([ox, oy])
                        else:  # this offspring is not significant
                            ctr += 1
                            LIP.append([ox, oy])
                    
                    # Check if offspring have their own offspring (become Type B)
                    has_grandchildren = any(
                        len(get_offspring(ox, oy, level, bandsize, H, W)) > 0
                        for ox, oy in offspring
                    )
                    
                    if has_grandchildren:
                        LIS_list[idx] = [x, y, 1]  # Convert to Type B
                    else:
                        LIS_list.pop(idx)
                        idx -= 1
                else:
                    ctr += 1  # offspring not significant

            else:  # Type B: process grandchildren (offspring of offspring)
                if bitstream[ctr] == 1:  # grandchildren are significant
                    ctr += 1
                    offspring = get_offspring(x, y, level, bandsize, H, W)
                    
                    # Add each offspring as a new Type A entry
                    for ox, oy in offspring:
                        if 0 <= ox < H and 0 <= oy < W:
                            LIS_list.append([ox, oy, 0])
                    
                    LIS_list.pop(idx)
                    idx -= 1
                else:
                    ctr += 1  # grandchildren not significant
            
            idx += 1

        LIS = LIS_list

        # ===== REFINEMENT PASS =====
        if n < n_max:
            for i in range(LSP_len_before):
                if ctr >= len(bitstream):
                    return m
                x, y = LSP[i]
                refine_bit = bitstream[ctr]; ctr += 1
                val_sign = np.sign(m[x, y])
                m[x, y] += (2**(n-1)) * val_sign if refine_bit == 1 else -(2**(n-1)) * val_sign

        n -= 1

    return m