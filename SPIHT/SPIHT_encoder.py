"""
SPIHT Encoder 
--------------------------------------------
Author: Evelyn Xie
Revised by: Rui Shen, Claude
- Handles arbitrary (non-square) image sizes 
- Pads to next valid DWT multiple
- Initializes LIP/LIS/LSP following original SPIHT paper (Said & Pearlman, 1996)
- Adds integer headers [H, W, n_max, level]
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from dwt import runDWT, decodeDWT


# Padding helpers

def pad_to_multiple(img, k, mode="edge"):
    H, W = img.shape[:2]
    pad_h = (k - (H % k)) % k
    pad_w = (k - (W % k)) % k
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded, (pad_h, pad_w)


def unpad(img, pad_hw):
    pad_h, pad_w = pad_hw
    H, W = img.shape[:2]
    return img[:H - pad_h, :W - pad_w]


# Descendant search - adapted for DWT pyramid layout
def func_MyDescendant(x, y, set_type, m, level):
    """
    Compute descendants for multi-level DWT pyramid layout
    
    For a pixel at (x,y) in the current LL band:
    - Children are in the next-level subbands at the same relative position
    - Each level's subbands are arranged: LL (top-left), HL (top-right), 
      LH (bottom-left), HH (bottom-right)
    
    Parameters:
    -----------
    x, y : int
        Coordinates in the LL band of the current decomposition level
    set_type : int
        0 = Type A (include direct children + all descendants)
        1 = Type B (exclude direct children, only grandchildren and deeper)
    m : np.ndarray
        Full wavelet coefficient matrix (H×W)
    level : int
        DWT decomposition level (how many times DWT was applied)
    
    Returns:
    --------
    float : Maximum absolute value among descendants
    """
    x, y = int(x), int(y)
    H, W = m.shape
    max_vals = []
    
    # Size of the finest LL band
    bandsize = H // (2 ** level)
    
    # For a pixel at (x,y) in LL band (size: bandsize × bandsize),
    # its 4 children are at the same (x,y) position in the 4 subbands:
    # - LL → (x, y) - stays in LL for next decomposition
    # - HL → (x, y + bandsize)
    # - LH → (x + bandsize, y)
    # - HH → (x + bandsize, y + bandsize)
    
    offspring_positions = [
        (x, y),                      # Child in LL (may have further descendants)
        (x, y + bandsize),           # Child in HL
        (x + bandsize, y),           # Child in LH
        (x + bandsize, y + bandsize) # Child in HH
    ]
    
    if set_type == 0:
        # Type A: include direct children + all deeper descendants
        for ox, oy in offspring_positions:
            if ox < H and oy < W:
                max_vals.append(abs(m[ox, oy]))
                
                # Recurse only for the LL child (only one with descendants)
                if ox == x and oy == y:
                    next_bandsize = bandsize // 2
                    if next_bandsize >= 1:
                        desc_max = func_MyDescendant_recursive(ox, oy, 0, m, level, next_bandsize)
                        max_vals.append(desc_max)
    
    else:
        # Type B: exclude direct children, test only grandchildren and deeper
        ox, oy = x, y  # The LL child position
        if ox < H and oy < W:
            next_bandsize = bandsize // 2
            if next_bandsize >= 1:
                # Test grandchildren
                grandchild_positions = [
                    (ox, oy),
                    (ox, oy + next_bandsize),
                    (ox + next_bandsize, oy),
                    (ox + next_bandsize, oy + next_bandsize)
                ]
                for gx, gy in grandchild_positions:
                    if gx < H and gy < W:
                        max_vals.append(abs(m[gx, gy]))
                        # Recurse for LL grandchild
                        if gx == ox and gy == oy:
                            next_next_bandsize = next_bandsize // 2
                            if next_next_bandsize >= 1:
                                desc_max = func_MyDescendant_recursive(gx, gy, 0, m, level, next_next_bandsize)
                                max_vals.append(desc_max)
    
    return max(max_vals) if max_vals else 0


def func_MyDescendant_recursive(x, y, set_type, m, total_level, current_bandsize):
    """
    Recursive helper that tracks the current bandsize through decomposition levels
    
    Parameters:
    -----------
    x, y : int
        Position in current LL band
    set_type : int
        0 = Type A, 1 = Type B
    m : np.ndarray
        Full coefficient matrix
    total_level : int
        Total DWT decomposition level
    current_bandsize : int
        Size of the current LL band being processed
    """
    x, y = int(x), int(y)
    H, W = m.shape
    max_vals = []
    
    offspring_positions = [
        (x, y),
        (x, y + current_bandsize),
        (x + current_bandsize, y),
        (x + current_bandsize, y + current_bandsize)
    ]
    
    if set_type == 0:
        for ox, oy in offspring_positions:
            if ox < H and oy < W:
                max_vals.append(abs(m[ox, oy]))
                
                if ox == x and oy == y:
                    next_bandsize = current_bandsize // 2
                    if next_bandsize >= 1:
                        desc_max = func_MyDescendant_recursive(ox, oy, 0, m, total_level, next_bandsize)
                        max_vals.append(desc_max)
    else:
        # Type B
        ox, oy = x, y
        if ox < H and oy < W:
            next_bandsize = current_bandsize // 2
            if next_bandsize >= 1:
                grandchild_positions = [
                    (ox, oy),
                    (ox, oy + next_bandsize),
                    (ox + next_bandsize, oy),
                    (ox + next_bandsize, oy + next_bandsize)
                ]
                for gx, gy in grandchild_positions:
                    if gx < H and gy < W:
                        max_vals.append(abs(m[gx, gy]))
                        if gx == ox and gy == oy:
                            next_next_bandsize = next_bandsize // 2
                            if next_next_bandsize >= 1:
                                desc_max = func_MyDescendant_recursive(gx, gy, 0, m, total_level, next_next_bandsize)
                                max_vals.append(desc_max)
    
    return max(max_vals) if max_vals else 0


# Initialization of LIP / LIS / LSP
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
    
    return np.array(LIP, dtype=np.int32), np.array(LIS, dtype=np.int32), []


def func_MySPIHT_Enc(m, max_bits=4096, level=1):
    """
    SPIHT Encoder

    Parameters
    ----------
    m : np.ndarray
        Wavelet coefficient matrix (after DWT)
    max_bits : int
        Maximum number of bits in output buffer
    level : int
        DWT decomposition level

    Returns
    -------
    np.ndarray
        Encoded integer bitstream (first 4 entries are header)
    """
    out = 2 * np.ones(max_bits, dtype=np.int32)
    index = 0
    bitctr = 0

    max_val = np.abs(m).max()
    n_max = int(np.floor(np.log2(max_val))) if max_val > 0 else 0
    n = n_max

    H, W = m.shape
    LIP, LIS, LSP = init_spiht_lists(m, level)
   
    # ---------- HEADER ----------
    out[0] = H
    out[1] = W
    out[2] = n_max
    out[3] = level
    index = 4
    bitctr = 0

    # ==========================================================
    # MAIN ENCODING LOOP
    # ==========================================================
    while index < max_bits and n >= 0:

        # Record number of significant coefficients before this pass
        LSP_len_before = len(LSP)
        LIP_remove = []

        # ---- SORTING PASS: LIP ----
        for k in range(len(LIP)):
            if index >= max_bits:
                return out[:index]
            x, y = LIP[k]
            if abs(m[x, y]) >= 2**n:
                out[index] = 1; index += 1; bitctr += 1
                if index >= max_bits: return out[:index]
                sign = 1 if m[x, y] >= 0 else 0
                out[index] = sign; index += 1; bitctr += 1
                if index >= max_bits: return out[:index]
                LSP.append([x, y])
                LIP_remove.append(k)
            else:
                out[index] = 0; index += 1; bitctr += 1

        if LIP_remove:
            LIP = np.delete(LIP, LIP_remove, axis=0)

        # ---- SORTING PASS: LIS (with dynamic growth for Type B offspring) ----
        
        # Convert to list for dynamic growth during iteration
        LIS_list = LIS.tolist() if isinstance(LIS, np.ndarray) else list(LIS)
        idx = 0
        
        while idx < len(LIS_list):
            if index >= max_bits:
                return out[:index]
            
            entry = LIS_list[idx]
            x, y, typ = entry
            
            if typ == 0:  # Type A
                max_d = func_MyDescendant(x, y, 0, m, level)
                if max_d >= 2**n:
                    out[index] = 1; index += 1; bitctr += 1
                    if index >= max_bits: return out[:index]
                    
                    # Get offspring positions based on DWT layout
                    bandsize = H // (2 ** level)
                    offspring = [
                        (x, y),
                        (x, y + bandsize),
                        (x + bandsize, y),
                        (x + bandsize, y + bandsize)
                    ]
                    
                    for ox, oy in offspring:
                        if ox < H and oy < W:
                            if index >= max_bits: return out[:index]
                            if abs(m[ox, oy]) >= 2**n:
                                out[index] = 1; index += 1; bitctr += 1
                                if index >= max_bits: return out[:index]
                                sign = 1 if m[ox, oy] >= 0 else 0
                                out[index] = sign; index += 1; bitctr += 1
                                if index >= max_bits: return out[:index]
                                LSP.append([ox, oy])
                            else:
                                out[index] = 0; index += 1; bitctr += 1
                                if len(LIP) > 0:
                                    LIP = np.vstack([LIP, [ox, oy]])
                                else:
                                    LIP = np.array([[ox, oy]], dtype=np.int32)
                    
                    # Convert to Type B if grandchildren exist
                    # Check if the LL child has descendants
                    next_bandsize = bandsize // 2
                    if next_bandsize >= 1:
                        LIS_list[idx] = [x, y, 1]  # Update in place to Type B
                    else:
                        # No grandchildren, remove from LIS
                        LIS_list.pop(idx)
                        idx -= 1  # Adjust index since we removed an element
                else:
                    out[index] = 0; index += 1; bitctr += 1
                    # Keep as Type A, no change needed
            
            else:  # Type B
                max_d = func_MyDescendant(x, y, 1, m, level)
                if max_d >= 2**n:
                    out[index] = 1; index += 1; bitctr += 1
                    if index >= max_bits: return out[:index]
                    
                    # Add grandchildren (which are children of the LL child)
                    bandsize = H // (2 ** level)
                    next_bandsize = bandsize // 2
                    
                    # The LL child is at (x,y), its children are:
                    offspring = [
                        (x, y),
                        (x, y + next_bandsize),
                        (x + next_bandsize, y),
                        (x + next_bandsize, y + next_bandsize)
                    ]
                    
                    # Add offspring to END of LIS (processed in same bitplane)
                    for ox, oy in offspring:
                        if ox < H and oy < W:
                            LIS_list.append([ox, oy, 0])
                    
                    # Remove this Type B entry (fully processed)
                    LIS_list.pop(idx)
                    idx -= 1  # Adjust index since we removed an element
                else:
                    out[index] = 0; index += 1; bitctr += 1
                    # Keep as Type B for next bitplane, no change needed
            
            idx += 1
        
        # Convert back to numpy array for next bitplane
        LIS = np.array(LIS_list, dtype=np.int32) if LIS_list else np.array([], dtype=np.int32).reshape(0, 3)

        # ---- REFINEMENT PASS ----
        if n < n_max:
            for i in range(LSP_len_before):
                if index >= max_bits:
                    return out[:index]
                x, y = LSP[i]
                val = int(abs(m[x, y]))
                bit = (val >> n) & 1  # output n-th most significant bit
                out[index] = bit; index += 1; bitctr += 1

        # Move to next bitplane
        n -= 1

    return out[:index]