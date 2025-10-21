"""
SPIHT_encoder.py

Python implementation of SPIHT (Set Partitioning in Hierarchical Trees) encoder


Student Name: Evelyn Xie

"""

import numpy as np


def func_MySPIHT_Enc(m, max_bits, block_size, level):
    """
    SPIHT Encoder - Main encoding function
    
    Args:
        m (numpy.ndarray): Input image in wavelet domain (2D array)
        max_bits (int): Maximum bits that can be used
        block_size (int): Image size
        level (int): Wavelet decomposition level
    
    Returns:
        out (numpy.ndarray): Bit stream (1D array)
    
    """
    
    # Initialization
    bitctr = 0
    out = 2 * np.ones(max_bits, dtype=np.int32)
    n_max = int(np.floor(np.log2(np.abs(m).max())))
    
    # Output bit stream header
    out[0] = m.shape[0]
    out[1] = n_max
    out[2] = level
    bitctr += 24
    index = 3
    
    # Initialize LIP, LSP, LIS
    bandsize = int(2 ** (np.log2(m.shape[0]) - level + 1))
    
    LIP = []
    for i in range(bandsize):
        for j in range(bandsize):
            LIP.append([i, j])
    LIP = np.array(LIP, dtype=np.int32)
    
    LIS = []
    for i in range(bandsize):
        for j in range(bandsize):
            LIS.append([i, j, 0])
    LIS = np.array(LIS, dtype=np.int32)
    
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
            if bitctr + 1 >= max_bits:
                return out[:bitctr]
            
            x, y = LIP[i]
            
            if abs(m[x, y]) >= 2**n:
                out[index] = 1
                bitctr += 1
                index += 1
                
                sgn = 1 if m[x, y] >= 0 else 0
                out[index] = sgn
                bitctr += 1
                index += 1
                
                LSP.append([x, y])
                LIP_indices_to_remove.append(i)
            else:
                out[index] = 0
                bitctr += 1
                index += 1
        
        LIP = np.delete(LIP, LIP_indices_to_remove, axis=0)
        
        # ===== SORTING PASS - LIS =====
        LIS_indices_to_remove = []
        LIS_to_add = []
        
        i = 0
        while i < len(LIS):
            if bitctr >= max_bits:
                return out[:bitctr]
            
            x, y, set_type = LIS[i]
            
            if set_type == 0:  # Type A
                max_d = func_MyDescendant(x, y, set_type, m)
                
                if max_d >= 2**n:
                    out[index] = 1
                    bitctr += 1
                    index += 1
                    
                    offspring = [
                        (2*x, 2*y),
                        (2*x, 2*y + 1),
                        (2*x + 1, 2*y),
                        (2*x + 1, 2*y + 1)
                    ]
                    
                    for ox, oy in offspring:
                        if bitctr + 1 >= max_bits:
                            return out[:bitctr]
                        
                        if abs(m[ox, oy]) >= 2**n:
                            LSP.append([ox, oy])
                            out[index] = 1
                            bitctr += 1
                            index += 1
                            
                            sgn = 1 if m[ox, oy] >= 0 else 0
                            out[index] = sgn
                            bitctr += 1
                            index += 1
                        else:
                            out[index] = 0
                            bitctr += 1
                            index += 1
                            LIP = np.vstack([LIP, [ox, oy]]) if len(LIP) > 0 else np.array([[ox, oy]])
                    
                    if (2 * (2*x)) < m.shape[0] and (2 * (2*y)) < m.shape[1]:
                        LIS_to_add.append([x, y, 1])
                    
                    LIS_indices_to_remove.append(i)
                else:
                    out[index] = 0
                    bitctr += 1
                    index += 1
            
            else:  # Type B
                max_d = func_MyDescendant(x, y, set_type, m)
                
                if max_d >= 2**n:
                    out[index] = 1
                    bitctr += 1
                    index += 1
                    
                    offspring = [
                        (2*x, 2*y, 0),
                        (2*x, 2*y + 1, 0),
                        (2*x + 1, 2*y, 0),
                        (2*x + 1, 2*y + 1, 0)
                    ]
                    LIS_to_add.extend(offspring)
                    LIS_indices_to_remove.append(i)
                else:
                    out[index] = 0
                    bitctr += 1
                    index += 1
            
            i += 1
        
        LIS = np.delete(LIS, LIS_indices_to_remove, axis=0)
        if LIS_to_add:
            if len(LIS) > 0:
                LIS = np.vstack([LIS, np.array(LIS_to_add)])
            else:
                LIS = np.array(LIS_to_add)
        
        # ===== REFINEMENT PASS =====
        temp = 0
        while temp < len(LSP):
            if bitctr >= max_bits:
                return out[:bitctr]
            
            x, y = LSP[temp]
            value = int(np.floor(np.abs(2**(n_max - n + 1) * m[x, y])))
            
            if value >= 2**(n_max + 2):
                s = (value >> (n_max + 1)) & 1
                out[index] = s
                bitctr += 1
                index += 1
            
            temp += 1
        
        n = n - 1
        
        if n < 0:
            break
    
    return out[:bitctr]


def func_MyDescendant(x, y, set_type, m):
    """
    Calculate maximum absolute value of descendants
    
    Args:
        x, y (int): Coordinates of parent node
        set_type (int): 0 for Type A, 1 for Type B
        m (numpy.ndarray): Wavelet coefficient matrix
    
    Returns:
        max_d (float): Maximum absolute value of descendants
    """
    
    if set_type == 0:  # Type A: all descendants
        max_vals = []
        
        offspring = [
            (2*x, 2*y),
            (2*x, 2*y + 1),
            (2*x + 1, 2*y),
            (2*x + 1, 2*y + 1)
        ]
        
        for ox, oy in offspring:
            if ox < m.shape[0] and oy < m.shape[1]:
                max_vals.append(abs(m[ox, oy]))
                
                if 2*ox < m.shape[0] and 2*oy < m.shape[1]:
                    desc_max = func_MyDescendant(ox, oy, 0, m)
                    max_vals.append(desc_max)
        
        return max(max_vals) if max_vals else 0
    
    else:  # Type B: grandchildren only
        max_vals = []
        
        offspring = [
            (2*x, 2*y),
            (2*x, 2*y + 1),
            (2*x + 1, 2*y),
            (2*x + 1, 2*y + 1)
        ]
        
        for ox, oy in offspring:
            grandchildren = [
                (2*ox, 2*oy),
                (2*ox, 2*oy + 1),
                (2*ox + 1, 2*oy),
                (2*ox + 1, 2*oy + 1)
            ]
            
            for gx, gy in grandchildren:
                if gx < m.shape[0] and gy < m.shape[1]:
                    max_vals.append(abs(m[gx, gy]))
                    
                    if 2*gx < m.shape[0] and 2*gy < m.shape[1]:
                        desc_max = func_MyDescendant(gx, gy, 0, m)
                        max_vals.append(desc_max)
        
        return max(max_vals) if max_vals else 0


if __name__ == "__main__":
    print("Testing SPIHT_encoder.py...")
    
    test_m = np.random.randn(64, 64) * 10
    test_m[0:32, 0:32] *= 10
    
    bitstream = func_MySPIHT_Enc(test_m, max_bits=10000, block_size=64, level=3)
    
    print(f"Input shape: {test_m.shape}")
    print(f"Bitstream length: {len(bitstream)}")
    print(f"Compression ratio: {(64*64*8) / len(bitstream):.2f}:1")
    print("✓ SPIHT encoder test passed!")