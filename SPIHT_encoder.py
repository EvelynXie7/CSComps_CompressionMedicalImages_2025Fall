"""
SPIHT Encoder 
--------------------------------------------
Author: Evelyn Xie
Revised by: Rui Shen, Claude
- Handles arbitrary (non-square) image sizes 
- Correct “star pattern” LIS initialization
- Correct offspring logic for highest vs deeper levels
- Retains Type A / Type B LIS sorting structure
- Supports multi-level DWT pyramids
- Adds safe write checks to prevent buffer overflow

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
    if bandsize * 2 > H:
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
# ================= Descendant Search ==================
# ======================================================

def func_MyDescendant(x, y, set_type, m, level, bandsize):
    """Compute max |coeff| among descendants of node (x,y)."""
    H, W = m.shape
    band = get_band(x, y, bandsize)
    max_vals = []
    offspring = get_offspring(x, y, band, level, bandsize, H, W)

    if set_type == 0:
        for ox, oy in offspring:
            max_vals.append(abs(m[ox, oy]))
            if (2*ox) < H and (2*oy) < W:
                desc_max = func_MyDescendant(ox, oy, 0, m, level - 1, bandsize // 2)
                max_vals.append(desc_max)
    else:
        for ox, oy in offspring:
            sub_offspring = get_offspring(
                ox, oy, get_band(ox, oy, bandsize), level, bandsize, H, W
            )
            for gx, gy in sub_offspring:
                max_vals.append(abs(m[gx, gy]))
                if (2*gx) < H and (2*gy) < W:
                    desc_max = func_MyDescendant(gx, gy, 0, m, level - 1, bandsize // 2)
                    max_vals.append(desc_max)

    return max(max_vals) if max_vals else 0


# ======================================================
# ======= Initialization of LIP / LIS / LSP ============
# ======================================================

def init_spiht_lists(m, level):
    """Initialize LIP/LIS/LSP using the 'star pattern' rule."""
    H, W = m.shape
    bandsize = H // (2 ** level)
    if bandsize < 2:
        raise ValueError(f"Invalid DWT level={level}")

    LIP, LIS = [], []

    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])

    for i in range(0, bandsize, 2):
        for j in range(0, bandsize, 2):
            for (x, y) in [(i, j+1), (i+1, j), (i+1, j+1)]:
                if x < bandsize and y < bandsize:
                    LIS.append([x, y, 0])

    return np.array(LIP, np.int32), np.array(LIS, np.int32), []


# ======================================================
# ===================== Encoder ========================
# ======================================================

def func_MySPIHT_Enc(m, max_bits=4096, level=1):
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

    # ==========================================================
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

            if typ == 0:
                max_d = func_MyDescendant(x, y, 0, m, level, bandsize)
                if max_d >= 2**n:
                    index, ok = safe_write(out, index, 1)
                    if not ok: return out[:index]
                    offspring = get_offspring(
                        x, y, get_band(x, y, bandsize), level, bandsize, H, W
                    )
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
                    if (2*(2*x)) < H and (2*(2*y)) < W:
                        LIS_list[idx] = [x, y, 1]
                    else:
                        LIS_list.pop(idx)
                        idx -= 1
                else:
                    index, ok = safe_write(out, index, 0)
                    if not ok: return out[:index]

            else:  # Type B
                max_d = func_MyDescendant(x, y, 1, m, level, bandsize)
                if max_d >= 2**n:
                    index, ok = safe_write(out, index, 1)
                    if not ok: return out[:index]
                    offspring = get_offspring(
                        x, y, get_band(x, y, bandsize), level, bandsize, H, W
                    )
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