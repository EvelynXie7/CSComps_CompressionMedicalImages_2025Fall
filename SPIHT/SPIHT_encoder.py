"""
SPIHT Encoder 
--------------------------------------------
Author: Evelyn Xie
Revised by: Rui Shen, Claude
- Handles arbitrary (non-square) image sizes 
- Correct "star pattern" LIS initialization
- Correct offspring logic for deepest LL vs high-frequency bands
- Retains Type A / Type B LIS sorting structure
- Supports multi-level DWT pyramids
- Adds safe write checks to prevent buffer overflow
- Clearer naming: is_in_deepest_LL, is_LL_root
- Fixed: bandsize parameter is ALWAYS deepest LL bandsize, never changes

- Pads to next valid DWT multiple
- Initializes LIP/LIS/LSP
- Adds integer headers [H, W, n_max, level]
- Fixed Type B processing to handle offspring in same bitplane
"""

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from dwt import runDWT


# ======================================================
# ================ Padding Utilities ===================
# ======================================================

def pad_to_multiple(img, k, mode="edge"):
    """Pad an image so that H,W are multiples of k."""
    H, W = img.shape[:2]
    pad_h = (k - (H % k)) % k
    pad_w = (k - (W % k)) % k
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded, (pad_h, pad_w)


def unpad(img, pad_hw):
    """Remove padding added by pad_to_multiple."""
    pad_h, pad_w = pad_hw
    H, W = img.shape[:2]
    return img[:H - pad_h, :W - pad_w]


# ======================================================
# ============ Safe Write Helper =======================
# ======================================================

def safe_write(out, index, val):
    """Safely write one integer to output buffer, preventing overflow."""
    if index >= len(out):
        print(f"[Warning] Buffer overflow at index={index}/{len(out)}. Truncating output.")
        return index, False
    out[index] = val
    return index + 1, True


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
    Simple rule: offspring are at (2*x, 2*y) with 4-child expansion
    """
    # Case 1: LL roots have no offspring
    if is_LL_root(x, y, bandsize):
        return []
    
    # Case 2: Non-root LL coefficients → offspring in HL/LH/HH at same level
    if is_in_deepest_LL(x, y, bandsize):
        block_x, block_y = x // 2, y // 2
        in_block_x, in_block_y = x % 2, y % 2
        
        offset = bandsize
        
        if in_block_x == 0 and in_block_y == 1:  # (0,1) → LH
            base_x, base_y = 2*block_x, 2*block_y + offset
        elif in_block_x == 1 and in_block_y == 0:  # (1,0) → HL
            base_x, base_y = 2*block_x + offset, 2*block_y
        elif in_block_x == 1 and in_block_y == 1:  # (1,1) → HH
            base_x, base_y = 2*block_x + offset, 2*block_y + offset
        else:
            return []
        
        offspring = [
            (base_x, base_y), (base_x, base_y + 1),
            (base_x + 1, base_y), (base_x + 1, base_y + 1)
        ]
        
        return [(ox, oy) for ox, oy in offspring if ox < H and oy < W]
    
    # Case 3: HL/LH/HH coefficients → just double the coordinates!
    base_x, base_y = 2 * x, 2 * y
    
    # Check if offspring would be out of bounds
    if base_x >= H or base_y >= W:
        return []
    
    offspring = [
        (base_x, base_y), (base_x, base_y + 1),
        (base_x + 1, base_y), (base_x + 1, base_y + 1)
    ]
    
    return [(ox, oy) for ox, oy in offspring if ox < H and oy < W]


# ======================================================
# ================= Descendant Search ==================
# ======================================================

def func_MyDescendant(x, y, set_type, m, level, bandsize):
    """
    Compute max |coeff| among descendants of node (x,y).
    
    Parameters:
    - x, y: current node coordinates
    - set_type: 0 for Type A (direct offspring), 1 for Type B (grandchildren)
    - m: DWT coefficient matrix
    - level: DWT decomposition level (fixed, never changes)
    - bandsize: size of deepest LL band (ALWAYS deepest level, never changes)
    """
    H, W = m.shape
    max_vals = []
    offspring = get_offspring(x, y, level, bandsize, H, W)

    if set_type == 0:  # Type A: check direct offspring and their descendants
        for ox, oy in offspring:
            max_vals.append(abs(m[ox, oy]))
            
            # Recurse to check descendants of offspring
            # Use SAME bandsize (always deepest LL bandsize)
            if len(get_offspring(ox, oy, level, bandsize, H, W)) > 0:
                desc_max = func_MyDescendant(ox, oy, 0, m, level, bandsize)
                max_vals.append(desc_max)
                
    else:  # Type B: check grandchildren (offspring of offspring)
        for ox, oy in offspring:
            # Get offspring of offspring (grandchildren)
            sub_offspring = get_offspring(ox, oy, level, bandsize, H, W)
            
            for gx, gy in sub_offspring:
                max_vals.append(abs(m[gx, gy]))
                # Recurse to check descendants of grandchildren
                # Use SAME bandsize (always deepest LL bandsize)
                if len(get_offspring(gx, gy, level, bandsize, H, W)) > 0:
                    desc_max = func_MyDescendant(gx, gy, 0, m, level, bandsize)
                    max_vals.append(desc_max)

    return max(max_vals) if max_vals else 0


# ======================================================
# ======= Initialization of LIP / LIS / LSP ============
# ======================================================

def init_spiht_lists(m, level):
    """
    Initialize LIP/LIS/LSP using the 'star pattern' rule.
    
    LIP: all coefficients in deepest LL band
    LIS: non-root coefficients in deepest LL (the 3 non-top-left in each 2×2 block)
         These are Type A entries that have offspring in HL/LH/HH
    LSP: empty initially
    """
    H, W = m.shape
    bandsize = H // (2 ** level)
    if bandsize < 2:
        raise ValueError(f"Invalid DWT level={level} for image size {H}×{W}")

    LIP, LIS = [], []

    # All deepest LL coefficients go in LIP
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # Non-root LL coefficients go in LIS (Type A)
    # These are the 3 non-top-left positions in each 2×2 block
    for i in range(0, bandsize, 2):
        for j in range(0, bandsize, 2):
            for (x, y) in [(i, j+1), (i+1, j), (i+1, j+1)]:
                if x < bandsize and y < bandsize:
                    LIS.append([x, y, 0])  # Type A

    return np.array(LIP, np.int32), np.array(LIS, np.int32), []


# ======================================================
# ===================== Encoder ========================
# ======================================================

def func_MySPIHT_Enc(m, max_bits=4096, level=1):
    """
    SPIHT Encoder with correct spatial-orientation tree structure.
    
    Parameters:
    - m: DWT coefficient matrix (output of runDWT)
    - max_bits: maximum output buffer size
    - level: DWT decomposition level
    
    Returns:
    - Encoded bitstream as numpy array
    """
    out = 2 * np.ones(max_bits, np.int32)
    index = 0

    H, W = m.shape
    max_val = np.abs(m).max()
    n_max = int(np.floor(np.log2(max_val))) if max_val > 0 else 0
    n = n_max
    bandsize = H // (2 ** level)

    LIP, LIS, LSP = init_spiht_lists(m, level)

    # ----- HEADER -----
    for val in [H, W, n_max, level]:
        index, ok = safe_write(out, index, val)
        if not ok: return out[:index]

    # ===== MAIN ENCODING LOOP =====
    while index < len(out) and n >= 0:
        LSP_len_before = len(LSP)
        LIP_remove = []

        # ---- SORTING PASS: LIP ----
        for k, (x, y) in enumerate(LIP):
            if abs(m[x, y]) >= 2**n:
                index, ok = safe_write(out, index, 1)
                if not ok: return out[:index]
                sign = 1 if m[x, y] >= 0 else 0
                index, ok = safe_write(out, index, sign)
                if not ok: return out[:index]
                LSP.append([x, y])
                LIP_remove.append(k)
            else:
                index, ok = safe_write(out, index, 0)
                if not ok: return out[:index]

        if LIP_remove:
            LIP = np.delete(LIP, LIP_remove, axis=0)

        # ---- SORTING PASS: LIS ----
        LIS_list = LIS.tolist() if isinstance(LIS, np.ndarray) else list(LIS)
        idx = 0
        while idx < len(LIS_list):
            x, y, typ = LIS_list[idx]

            if typ == 0:  # Type A: process direct offspring
                max_d = func_MyDescendant(x, y, 0, m, level, bandsize)
                if max_d >= 2**n:
                    index, ok = safe_write(out, index, 1)
                    if not ok: return out[:index]
                    
                    offspring = get_offspring(x, y, level, bandsize, H, W)
                    for ox, oy in offspring:
                        if abs(m[ox, oy]) >= 2**n:
                            index, ok = safe_write(out, index, 1)
                            if not ok: return out[:index]
                            sign = 1 if m[ox, oy] >= 0 else 0
                            index, ok = safe_write(out, index, sign)
                            if not ok: return out[:index]
                            LSP.append([ox, oy])
                        else:
                            index, ok = safe_write(out, index, 0)
                            if not ok: return out[:index]
                            LIP = np.vstack([LIP, [ox, oy]]) if len(LIP) else np.array([[ox, oy]], np.int32)
                    
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
                    index, ok = safe_write(out, index, 0)
                    if not ok: return out[:index]

            else:  # Type B: process grandchildren (offspring of offspring)
                max_d = func_MyDescendant(x, y, 1, m, level, bandsize)
                if max_d >= 2**n:
                    index, ok = safe_write(out, index, 1)
                    if not ok: return out[:index]
                    
                    # Add each offspring as a new Type A entry
                    offspring = get_offspring(x, y, level, bandsize, H, W)
                    for ox, oy in offspring:
                        LIS_list.append([ox, oy, 0])
                    
                    LIS_list.pop(idx)
                    idx -= 1
                else:
                    index, ok = safe_write(out, index, 0)
                    if not ok: return out[:index]
            
            idx += 1

        LIS = np.array(LIS_list, np.int32) if LIS_list else np.empty((0, 3), np.int32)

        # ---- REFINEMENT PASS ----
        if n < n_max:
            for i in range(LSP_len_before):
                x, y = LSP[i]
                bit = (int(abs(m[x, y])) >> n) & 1
                index, ok = safe_write(out, index, bit)
                if not ok: return out[:index]

        n -= 1

    return out[:index]