"""
SPIHT Decoder 
--------------------------------------------
Author: Rui Shen
Revised by: Evelyn Xie, Claude
- Matches updated encoder logic
- Uses correct offspring structure (LL→HL/LH/HH)
- Supports multi-level DWT
- Maintains Type A/B LIS structure and star pattern init
"""

import numpy as np


# ======================================================
# ========== Band + Offspring Definitions ==============
# ======================================================

def get_band(x, y, bandsize):
    """Identify which subband a coefficient belongs to."""
    if x < bandsize and y < bandsize:
        return "LL"
    elif x < bandsize and y >= bandsize:
        return "LH"
    elif x >= bandsize and y < bandsize:
        return "HL"
    else:
        return "HH"


def get_offspring(x, y, band, level, bandsize, H, W):
    """
    Compute offspring coordinates according to spatial-orientation tree rules.
    - LL roots → children in HL/LH/HH of next finer level
    - LH/HL/HH → children in same-orientation subband below
    """
    if bandsize * 2 > H:  # no deeper level
        return []

    offspring = []
    offset = bandsize

    if band == "LL":
        offspring += [(2*x + offset, 2*y)]          # HL
        offspring += [(2*x,         2*y + offset)]  # LH
        offspring += [(2*x + offset, 2*y + offset)] # HH
    elif band == "HL":
        offspring += [(2*x + offset, 2*y)]
    elif band == "LH":
        offspring += [(2*x, 2*y + offset)]
    elif band == "HH":
        offspring += [(2*x + offset, 2*y + offset)]

    # Expand each offspring into 2×2 block
    expanded = []
    for ox, oy in offspring:
        expanded.extend([
            (ox, oy),
            (ox, oy + 1),
            (ox + 1, oy),
            (ox + 1, oy + 1)
        ])
    return [(ox, oy) for ox, oy in expanded if ox < H and oy < W]


# ======================================================
# ======= Initialization of LIP / LIS / LSP ============
# ======================================================

def init_spiht_lists(m, level):
    """
    Initialize LIP and LIS in star pattern matching encoder.
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
                    LIS.append([x, y, 0])

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
            band = get_band(x, y, bandsize)
            offspring = get_offspring(x, y, band, level, bandsize, H, W)

            if typ == 0:  # Type A
                if bitstream[ctr] == 1:
                    ctr += 1
                    for ox, oy in offspring:
                        if ctr >= len(bitstream): return m
                        if not (0 <= ox < H and 0 <= oy < W): continue
                        if bitstream[ctr] == 1:
                            ctr += 1
                            if ctr >= len(bitstream): return m
                            sign = bitstream[ctr]; ctr += 1
                            m[ox, oy] = (1.5 if sign == 1 else -1.5) * (2**n)
                            LSP.append([ox, oy])
                        else:
                            ctr += 1
                            LIP.append([ox, oy])
                    # check for grandchildren
                    if (2*(2*x) < H) and (2*(2*y) < W):
                        LIS_list[idx] = [x, y, 1]
                    else:
                        LIS_list.pop(idx)
                        idx -= 1
                else:
                    ctr += 1  # not significant

            else:  # Type B
                if bitstream[ctr] == 1:
                    ctr += 1
                    for ox, oy in offspring:
                        if 0 <= ox < H and 0 <= oy < W:
                            LIS_list.append([ox, oy, 0])
                    LIS_list.pop(idx)
                    idx -= 1
                else:
                    ctr += 1
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
