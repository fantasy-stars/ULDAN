import numpy as np

def decoder_bit2gray(src): # src 0-255
    ori_w, ori_h=src.shape[0],src.shape[1]
    src=np.array(src,dtype=np.uint8)

    b = np.zeros((ori_w//2, ori_h//4), dtype=np.uint8)

    for i in range(ori_w//2):
        for j in range(ori_h//4):
            row_start = i * 2
            col_start = j * 4
            binary_sequence = src[row_start:row_start + 2, col_start:col_start + 4].flatten()
            decimal_value = int(''.join(map(str, binary_sequence)), 2)
            b[i, j] = decimal_value

    return b

