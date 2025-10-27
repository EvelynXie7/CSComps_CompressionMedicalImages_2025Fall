"""
SPIHT_encoder.py

Python implementation of SPIHT (Set Partitioning in Hierarchical Trees) encoder


Student Name: Evelyn Xie

"""

import numpy as np
from loading import *

# Import the new DWT functions
from dwt import runDWT, decodeDWT
from utils import *

def pad_to_multiple(img, k, mode="edge"):
    """
    Pad image so its height/width are multiples of k.
    Returns padded image and (pad_h, pad_w) used.
    """
    H, W = img.shape[:2]
    pad_h = (k - (H % k)) % k
    pad_w = (k - (W % k)) % k
    if pad_h == 0 and pad_w == 0:
        return img, (0, 0)
    padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode=mode)
    return padded, (pad_h, pad_w)


def unpad(img, pad_hw):
    """
    Remove the padding that was added previously.
    """
    pad_h, pad_w = pad_hw
    H, W = img.shape[:2]
    if pad_h == 0 and pad_w == 0:
        return img
    return img[:H - pad_h, :W - pad_w]



def func_MySPIHT_Enc(m, max_bits, block_size, level):
    """
    SPIHT Encoder - Main encoding function
    """

    # pad to make DWT-friendly size
    k = 2 ** level
    m_padded, pad_hw = pad_to_multiple(m, k, mode="edge")
    m = m_padded

    # Initialization
    bitctr = 0
    out = 2 * np.ones(max_bits, dtype=np.int32)
    max_val = np.abs(m).max()
    if max_val == 0:
        n_max = 0
    else:
        n_max = int(np.floor(np.log2(max_val)))

    out[0] = m.shape[0]
    out[1] = n_max
    out[2] = level
    bitctr += 24
    index = 3

    # Initialize LIP, LSP, LIS
    bandsize = int(2 ** (np.log2(m.shape[0]) - level + 1))
    LIP = np.array([[i, j] for i in range(bandsize) for j in range(bandsize)], dtype=np.int32)
    LIS = np.array([[i, j, 0] for i in range(bandsize) for j in range(bandsize)], dtype=np.int32)

    # Remove LL band from LIS
    pstart = 0
    pend = bandsize // 2
    indices_to_remove = []
    for i in range(bandsize // 2):
        indices_to_remove.extend(range(pstart, pend))
        pdel = pend - pstart
        pstart = pstart + bandsize - pdel
        pend = pend + bandsize - pdel
    LIS = np.delete(LIS, indices_to_remove, axis=0)

    LSP = []
    n = n_max

    # Main coding loop
    while bitctr < max_bits:
        # ===== SORTING PASS - LIP =====
        LIP_indices_to_remove = []
        for i in range(len(LIP)):
            x, y = map(int, LIP[i])
            if bitctr + 1 >= max_bits:
                return out[:bitctr]
            if abs(m[x, y]) >= 2**n:
                out[index] = 1; bitctr += 1; index += 1
                sgn = 1 if m[x, y] >= 0 else 0
                out[index] = sgn; bitctr += 1; index += 1
                LSP.append([x, y])
                LIP_indices_to_remove.append(i)
            else:
                out[index] = 0; bitctr += 1; index += 1
        if len(LIP_indices_to_remove) > 0:
            LIP = np.delete(LIP, LIP_indices_to_remove, axis=0)

        # ===== SORTING PASS - LIS =====
        LIS_indices_to_remove = []
        LIS_to_add = []

        i = 0
        while i < len(LIS):
            x, y, set_type = map(int, LIS[i])
            if bitctr >= max_bits:
                return out[:bitctr]

            if set_type == 0:
                max_d = func_MyDescendant(x, y, set_type, m)
                if max_d >= 2**n:
                    out[index] = 1; bitctr += 1; index += 1

                    offspring = [
                        (2*x, 2*y),
                        (2*x, 2*y + 1),
                        (2*x + 1, 2*y),
                        (2*x + 1, 2*y + 1)
                    ]

                    for ox, oy in offspring:
                        ox, oy = int(ox), int(oy)
                        if ox >= m.shape[0] or oy >= m.shape[1]:
                            continue
                        if abs(m[ox, oy]) >= 2**n:
                            LSP.append([ox, oy])
                            out[index] = 1; bitctr += 1; index += 1
                            sgn = 1 if m[ox, oy] >= 0 else 0
                            out[index] = sgn; bitctr += 1; index += 1
                        else:
                            out[index] = 0; bitctr += 1; index += 1
                            if len(LIP) > 0:
                                LIP = np.vstack([LIP, [ox, oy]])
                            else:
                                LIP = np.array([[ox, oy]], dtype=np.int32)

                    if (2*(2*x)) < m.shape[0] and (2*(2*y)) < m.shape[1]:
                        LIS_to_add.append([x, y, 1])

                    LIS_indices_to_remove.append(i)
                else:
                    out[index] = 0; bitctr += 1; index += 1

            else:  # Type B
                max_d = func_MyDescendant(x, y, set_type, m)
                if max_d >= 2**n:
                    out[index] = 1; bitctr += 1; index += 1
                    offspring = [
                        (2*x, 2*y, 0),
                        (2*x, 2*y + 1, 0),
                        (2*x + 1, 2*y, 0),
                        (2*x + 1, 2*y + 1, 0)
                    ]
                    LIS_to_add.extend(offspring)
                    LIS_indices_to_remove.append(i)
                else:
                    out[index] = 0; bitctr += 1; index += 1
            i += 1

        if len(LIS_indices_to_remove) > 0:
            LIS = np.delete(LIS, LIS_indices_to_remove, axis=0)
        if len(LIS_to_add) > 0:
            add_arr = np.array(LIS_to_add, dtype=np.int32)
            if len(LIS) > 0:
                LIS = np.vstack([LIS, add_arr])
            else:
                LIS = add_arr

        # ===== REFINEMENT PASS =====
        temp = 0
        while temp < len(LSP):
            x, y = map(int, LSP[temp])
            if bitctr >= max_bits:
                return out[:bitctr]
            value = int(np.floor(np.abs(2**(n_max - n + 1) * m[x, y])))
            if value >= 2**(n_max + 2):
                s = (value >> (n_max + 1)) & 1
                out[index] = s; bitctr += 1; index += 1
            temp += 1

        n -= 1
        if n < 0:
            break

    return out[:bitctr]


def func_MyDescendant(x, y, set_type, m):
    """
    Calculate maximum absolute value of descendants with safe integer indexing
    """
    x, y = int(x), int(y)
    max_vals = []

    offspring = [
        (2*x, 2*y),
        (2*x, 2*y + 1),
        (2*x + 1, 2*y),
        (2*x + 1, 2*y + 1)
    ]

    if set_type == 0:
        for ox, oy in offspring:
            ox, oy = int(ox), int(oy)
            if ox < m.shape[0] and oy < m.shape[1]:
                max_vals.append(abs(m[ox, oy]))
                if 2*ox < m.shape[0] and 2*oy < m.shape[1]:
                    desc_max = func_MyDescendant(ox, oy, 0, m)
                    max_vals.append(desc_max)
    else:
        for ox, oy in offspring:
            ox, oy = int(ox), int(oy)
            grandchildren = [
                (2*ox, 2*oy),
                (2*ox, 2*oy + 1),
                (2*ox + 1, 2*oy),
                (2*ox + 1, 2*oy + 1)
            ]
            for gx, gy in grandchildren:
                gx, gy = int(gx), int(gy)
                if gx < m.shape[0] and gy < m.shape[1]:
                    max_vals.append(abs(m[gx, gy]))
                    if 2*gx < m.shape[0] and 2*gy < m.shape[1]:
                        desc_max = func_MyDescendant(gx, gy, 0, m)
                        max_vals.append(desc_max)

    return max(max_vals) if max_vals else 0


