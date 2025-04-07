import random
import os
import numpy as np
import scipy
from scipy.io import savemat
from utils import read_h265_file,save_to_binary_file,numpy_to_bytes


if __name__=='__main__':
    mat_data = scipy.io.loadmat('xxx.mat')
    recover_data = mat_data['decoder_data_all'].flatten()
    recover_data = np.array(recover_data, dtype=np.int8)

    ori_npy_data=np.load('xxx.npy')
    recover_data=recover_data[:ori_npy_data.shape[0]]

    acc=np.mean(ori_npy_data.flatten()==recover_data.flatten())

    recover_bytes=numpy_to_bytes(recover_data)

    save_to_binary_file('xx.hevc', recover_bytes)
   
   