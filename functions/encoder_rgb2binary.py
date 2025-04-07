import numpy as np

def encoder_gray2bit(src):
    ori_size=src.shape[0]
    a_bin = np.unpackbits(src.astype(np.uint8)[:, :, np.newaxis], axis=2)

    b = np.zeros((ori_size*2, ori_size*4), dtype=np.uint8)

    for i in range(ori_size):
        for j in range(ori_size):
            row_start = i * 2
            col_start = j * 4
            b[row_start:row_start + 2, col_start:col_start + 4] = a_bin[i, j].reshape(2,4)

    return b

