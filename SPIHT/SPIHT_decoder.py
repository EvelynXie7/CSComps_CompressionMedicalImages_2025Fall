"""
SPIHT Decoder 
--------------------------------------------
Author: Rui Shen
Revised by: Evelyn Xie, Claude
- Matches updated encoder logic with correct spatial-orientation tree
- Uses correct offspring structure for deepest LL vs HL/LH/HH bands
- Supports multi-level DWT
- Maintains Type A/B LIS structure and star pattern init
- Clear helper functions: is_in_deepest_LL, is_LL_root
"""

import numpy as np


# ======================================================
# ========== Spatial-Orientation Tree Helpers ==========
# ======================================================

def is_in_deepest_LL(x, y, bandsize):
    """Check if coefficient is in the deepest LL region."""
    return x < bandsize and y < bandsize


def is_LL_root(x, y, bandsize):
    """
    Check if this is a 2×2 block root in deepest LL (top-left of each 2×2).
    These have NO offspring - they're leaves in the spatial-orientation tree.
    """
    return (x < bandsize and y < bandsize and 
            x % 2 == 0 and y % 2 == 0)

def get_offspring(x, y, level, bandsize, H, W):
    """
    Compute offspring coordinates according to spatial-orientation tree rules.
    
    Three cases:
    1. LL root (top-left of 2×2 in deepest LL): NO offspring
    2. Non-root LL coefficient (other 3 in each 2×2): offspring in HL/LH/HH at SAME level
    3. HL/LH/HH coefficients: offspring in same orientation at NEXT FINER level
    
    Parameters:
    - x, y: current coefficient coordinates in the full DWT array
    - level: DWT decomposition level
    - bandsize: size of deepest LL band (H // 2^level)
    - H, W: dimensions of the full DWT coefficient array
    """
    # Case 1: LL roots have no offspring
    if is_LL_root(x, y, bandsize):
        return []
    
    # Case 2: Non-root LL coefficients → offspring in HL/LH/HH at same level
    if is_in_deepest_LL(x, y, bandsize):
        block_x, block_y = x // 2, y // 2
        in_block_x, in_block_y = x % 2, y % 2
        
        offspring = []
        offset = bandsize
        
        # Map position in 2×2 block to corresponding high-freq band
        if in_block_x == 0 and in_block_y == 1:  # (0,1) → LH
            base_x, base_y = 2*block_x, 2*block_y + offset
        elif in_block_x == 1 and in_block_y == 0:  # (1,0) → HL
            base_x, base_y = 2*block_x + offset, 2*block_y
        elif in_block_x == 1 and in_block_y == 1:  # (1,1) → HH
            base_x, base_y = 2*block_x + offset, 2*block_y + offset
        else:
            return []  # (0,0) should have been caught by is_LL_root
        
        # 4 offspring in a 2×2 block
        offspring = [
            (base_x, base_y), (base_x, base_y + 1),
            (base_x + 1, base_y), (base_x + 1, base_y + 1)
        ]
        
        return [(ox, oy) for ox, oy in offspring if ox < H and oy < W]
    
    # Case 3: HL/LH/HH coefficients → offspring at next finer level
    # First, determine which level this coefficient is actually at
    # by checking which band region it falls into
    
    # Determine current level by finding which band region contains (x,y)
    current_level_bandsize = bandsize
    parent_level = level  # Start at deepest level
    
    # Move up levels until we find which band contains this coordinate
    while current_level_bandsize * 2 <= H:
        if (x < current_level_bandsize * 2 and y < current_level_bandsize * 2):
            # This coordinate is within the current level's extent
            break
        current_level_bandsize *= 2
        parent_level -= 1
    
    # Identify which high-freq band we're in at this level
    if x < current_level_bandsize and y >= current_level_bandsize:
        band = "LH"
    elif x >= current_level_bandsize and y < current_level_bandsize:
        band = "HL"
    elif x >= current_level_bandsize and y >= current_level_bandsize:
        band = "HH"
    else:
        return []
    
    # Check if offspring level exists
    offspring_bandsize = current_level_bandsize * 2
    if offspring_bandsize > H:
        return []
    
    # Convert to local coordinates within the current band
    if band == "HL":
        local_x = x - current_level_bandsize
        local_y = y
    elif band == "LH":
        local_x = x
        local_y = y - current_level_bandsize
    elif band == "HH":
        local_x = x - current_level_bandsize
        local_y = y - current_level_bandsize
    
    # Offspring are at 2× local position in the same orientation band at next level
    offspring_local_x = 2 * local_x
    offspring_local_y = 2 * local_y
    
    # Convert back to global coordinates
    if band == "HL":
        base_x = offspring_local_x + offspring_bandsize
        base_y = offspring_local_y
    elif band == "LH":
        base_x = offspring_local_x
        base_y = offspring_local_y + offspring_bandsize
    elif band == "HH":
        base_x = offspring_local_x + offspring_bandsize
        base_y = offspring_local_y + offspring_bandsize
    
    # Expand to 4 children in a 2×2 block
    offspring = [
        (base_x, base_y), (base_x, base_y + 1),
        (base_x + 1, base_y), (base_x + 1, base_y + 1)
    ]
    
    return [(ox, oy) for ox, oy in offspring if ox < H and oy < W]


# ======================================================
# ======= Initialization of LIP / LIS / LSP ============
# ======================================================

def init_spiht_lists(m, level):
    """
    Initialize LIP and LIS in star pattern matching encoder.
    
    LIP: all coefficients in deepest LL band
    LIS: non-root coefficients in deepest LL (the 3 non-top-left in each 2×2 block)
         These are Type A entries that have offspring in HL/LH/HH
    LSP: empty initially
    """
    H, W = m.shape
    bandsize = H // (2 ** level)
    if bandsize < 2:
        raise ValueError(f"Invalid DWT level={level}")

    LIP, LIS = [], []

    # LIP: all LL coefficients
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # LIS: star pattern (exclude top-left of each 2×2 LL block)
    for i in range(0, bandsize, 2):
        for j in range(0, bandsize, 2):
            for (x, y) in [(i, j+1), (i+1, j), (i+1, j+1)]:
                if x < bandsize and y < bandsize:
                    LIS.append([x, y, 0])  # Type A

    return LIP, LIS, []


# ======================================================
# ===================== Decoder ========================
# ======================================================

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
                    
                    # Check if this node has grandchildren to become Type B
                    has_grandchildren = False
                    if is_in_deepest_LL(x, y, bandsize):
                        # Offspring are in HL/LH/HH at same level
                        # They have grandchildren at next finer level
                        offspring_bandsize = bandsize * 2
                        if offspring_bandsize <= H and level > 1:
                            has_grandchildren = True
                    else:
                        # Offspring are at next finer level
                        # They have grandchildren at two levels finer
                        offspring_bandsize = bandsize * 2
                        if offspring_bandsize * 2 <= H and level > 1:
                            has_grandchildren = True
                    
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