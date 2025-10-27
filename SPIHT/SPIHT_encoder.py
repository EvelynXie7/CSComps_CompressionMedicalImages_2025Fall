"""
SPIHT Encoder 
--------------------------------------------
Author: Evelyn Xie

- Handles arbitrary (non-square) image sizes 
- Pads to next valid DWT multiple
- Initializes LIP/LIS/LSP
- Adds integer headers [H, W, n_max, level]

"""

import numpy as np
from dwt import runDWT, decodeDWT
from utils import *




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



# Descendant search
def func_MyDescendant(x, y, set_type, m):
    """
    Compute the maximum absolute value among the descendants of node (x, y)
    for use in SPIHT significance testing.

    Parameters
    ----------
    x, y : int
        Coordinates of the current node in the wavelet coefficient matrix.
    set_type : int
        0 = Type A (full descendant set D(x, y))
        1 = Type B (exclude direct children, test only L(x, y))
    m : np.ndarray
        Wavelet coefficient matrix.

    Returns
    -------
    float
        Maximum absolute coefficient value among the relevant descendant set.
    """
    x, y = int(x), int(y)
    H, W = m.shape
    max_vals = []

    # --- Direct offspring of (x, y) ---
    offspring = [
        (2*x, 2*y),
        (2*x, 2*y + 1),
        (2*x + 1, 2*y),
        (2*x + 1, 2*y + 1)
    ]

    if set_type == 0:
        # ---------- Type A: include children + all deeper descendants ----------
        for ox, oy in offspring:
            if ox < H and oy < W:
                # Include the direct child value
                max_vals.append(abs(m[ox, oy]))
                # Recurse into that child's descendants
                if (2*ox) < H and (2*oy) < W:
                    desc_max = func_MyDescendant(ox, oy, 0, m)
                    max_vals.append(desc_max)

    else:
        # ---------- Type B: exclude direct children, test only deeper descendants ----------
        for ox, oy in offspring:
            # For each child, check its own children (grandchildren of the original)
            grandchildren = [
                (2*ox, 2*oy),
                (2*ox, 2*oy + 1),
                (2*ox + 1, 2*oy),
                (2*ox + 1, 2*oy + 1)
            ]
            for gx, gy in grandchildren:
                if gx < H and gy < W:
                    max_vals.append(abs(m[gx, gy]))
                    # Recurse further if this grandchild has its own descendants
                    if (2*gx) < H and (2*gy) < W:
                        desc_max = func_MyDescendant(gx, gy, 0, m)
                        max_vals.append(desc_max)

    return max(max_vals) if max_vals else 0


# Initialization of LIP / LIS / LSP
def init_spiht_lists(m, level):
    H, W = m.shape
    bandsize = H // (2 ** level) 
    if bandsize == 1:
        raise ValueError(f"Invalid DWT level={level}: SPIHT requires LL ≥ 2×2.")
    LIP, LIS = [], []

    # LIP: all LL coefficients
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    # LIS: remove 1/4 top-left region in each bandsize/2 block
    if bandsize == 2:
        # For 2×2 LL, only (0,0) is removed
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
    return np.array(LIP, dtype=np.int32), np.array(LIS, dtype=np.int32), []


def func_MySPIHT_Enc(m, max_bits=4096, level=1):
    """
    SPIHT Encoder (Optimized + Correct Bitplane Handling)

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
    print(f"Initialized LIP={len(LIP)}, LIS={len(LIS)}, LL size={H//(2**level)}")

    # ---------- HEADER ----------
    out[0] = H
    out[1] = W
    out[2] = n_max
    out[3] = level
    index = 4
    bitctr = 0
    print(f"\nHeader written: [H={H}, W={W}, n_max={n_max}, level={level}]")

    # ==========================================================
    # MAIN ENCODING LOOP
    # ==========================================================
    while index < max_bits and n >= 0:
        print(f"\n=== Bitplane n={n} ===")

        # Record number of significant coefficients before this pass
        LSP_len_before = len(LSP)
        LIP_remove = []

        # ---- SORTING PASS: LIP ----
        for k in range(len(LIP)):
            if index >= max_bits:
                print("[Warning] Output buffer full.")
                return out[:index]
            x, y = LIP[k]
            if abs(m[x, y]) >= 2**n:
                out[index] = 1; index += 1; bitctr += 1
                if index >= max_bits: return out[:index]
                sign = 1 if m[x, y] >= 0 else 0
                out[index] = sign; index += 1; bitctr += 1
                LSP.append([x, y])
                LIP_remove.append(k)
            else:
                out[index] = 0; index += 1; bitctr += 1

        if LIP_remove:
            LIP = np.delete(LIP, LIP_remove, axis=0)

        # ---- SORTING PASS: LIS ----
        new_LIS = []
        for entry in LIS:
            if index >= max_bits:
                print("[Warning] Output buffer full.")
                return out[:index]
            x, y, typ = entry

            if typ == 0:  # Type A
                max_d = func_MyDescendant(x, y, 0, m)
                if max_d >= 2**n:
                    out[index] = 1; index += 1; bitctr += 1
                    offspring = [
                        (2*x, 2*y),
                        (2*x, 2*y + 1),
                        (2*x + 1, 2*y),
                        (2*x + 1, 2*y + 1)
                    ]
                    for ox, oy in offspring:
                        if ox < H and oy < W:
                            if abs(m[ox, oy]) >= 2**n:
                                out[index] = 1; index += 1; bitctr += 1
                                if index >= max_bits: return out[:index]
                                sign = 1 if m[ox, oy] >= 0 else 0
                                out[index] = sign; index += 1; bitctr += 1
                                LSP.append([ox, oy])
                                
                            else:
                                out[index] = 0; index += 1; bitctr += 1
                                if len(LIP) > 0:
                                    LIP = np.vstack([LIP, [ox, oy]])
                                else:
                                    LIP = np.array([[ox, oy]], dtype=np.int32)
        
                    if (2*(2*x)) < H and (2*(2*y)) < W:
                        new_LIS.append([x, y, 1])
                else:
                    out[index] = 0; index += 1; bitctr += 1
                    new_LIS.append([x, y, 0])
            else:  # Type B
                max_d = func_MyDescendant(x, y, 1, m)
                if max_d >= 2**n:
                    out[index] = 1; index += 1; bitctr += 1
                    offspring = [
                        (2*x, 2*y),
                        (2*x, 2*y + 1),
                        (2*x + 1, 2*y),
                        (2*x + 1, 2*y + 1)
                    ]
                    for ox, oy in offspring:
                        if ox < H and oy < W:
                            new_LIS.append([ox, oy, 0])
                else:
                    out[index] = 0; index += 1; bitctr += 1
                    new_LIS.append([x, y, 1])
        LIS = np.array(new_LIS, dtype=np.int32)

        # ---- REFINEMENT PASS ----
        if n < n_max:
            for i in range(LSP_len_before):
                x, y = LSP[i]
                if index >= max_bits:
                    print("[Warning] Output buffer full.")
                    return out[:index]
                val = int(abs(m[x, y]))
                bit = (val >> n) & 1  # output n-th most significant bit
                out[index] = bit; index += 1; bitctr += 1

        # Move to next bitplane
        n -= 1

    print(f"Encoding complete. Bits used (after header): {bitctr}")
    return out[:index]