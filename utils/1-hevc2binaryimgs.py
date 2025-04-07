import random
import os
import numpy as np
from scipy.io import savemat
from utils import read_h265_file,save_to_binary_file,numpy_to_bytes


if __name__=='__main__':
    H_265_file='crf28.hevc'
    file_content=read_h265_file(H_265_file)

    bit_string = ''.join(f'{byte:08b}' for byte in file_content)
    bit_vector = np.array([int(bit) for bit in bit_string], dtype=np.int8)

    np.save('vector_{}.npy'.format(bit_vector.shape[0]),bit_vector)
    savemat('vector_{}.mat'.format(bit_vector.shape[0]), {'array': bit_vector})  
