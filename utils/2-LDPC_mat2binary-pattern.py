import scipy.io
import numpy as np

mat_data = scipy.io.loadmat('xxx.mat')

array = mat_data['encoder_data_all'].flatten()

numpy_array = np.array(array, dtype=np.int8)

shape_x=64
pad_len=shape_x**2-(numpy_array.shape[0] % (shape_x**2))
print(pad_len)
