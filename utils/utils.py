import numpy as np
import scipy
import os

def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def cake_cutting_func(shape):
    N = int(np.log2(shape))
    x = scipy.linalg.hadamard(shape)
    a = int(np.sqrt(shape))

    cchdmm = np.zeros((shape, shape))
    num = np.zeros(shape)

    for i in range(shape):
        row = np.reshape(x[i, :], (a, a))
        num1 = 0
        num2 = 0
        for j in range(a-1):
            if row[0, j] != row[0, j+1]:
                num1 = num1 + 1
            if row[j, 0] != row[j+1, 0]:
                num2 = num2 + 1
        num[i] = (num1 + 1) * (num2 + 1)

    index = np.argsort(num)
    
    for k in range(shape):
        cchdmm[k, :] = x[index[k], :]
    

    return cchdmm

def gen_gray_code(N):
    sub_gray = np.array([[0], [1]], dtype=int)
    for n in range(2, N+1):
        top_gray = np.concatenate((np.zeros((2**(n-1), 1), dtype=int), sub_gray), axis=1)
        bottom_gray = np.concatenate((np.ones((2**(n-1), 1), dtype=int), sub_gray[::-1]), axis=1)
        sub_gray = np.concatenate((top_gray, bottom_gray), axis=0)
    return sub_gray
def walsh_func(shape):
    m = shape
    N = int(np.log2(m))
    x = scipy.linalg.hadamard(m)
    walsh = np.zeros((m, m))
    graycode = gen_gray_code(N)
    nh1 = np.zeros((m, N))

    for i in range(m):
        q = graycode[i, :]
        nh = 0
        for j in range(N-1, -1, -1):
            nh1[i, j] = q[j] * 2**(j)
        nh = np.sum(nh1[i, :])
        walsh[i, :] = x[int(nh), :]

    return walsh

def high_freq_func(shape):
    walshhdmm=walsh_func(shape)
    highfreqhdmm=np.zeros_like(walshhdmm)
    for i in range(shape):
        highfreqhdmm[i,:]=walshhdmm[shape-1-i,:]
    return highfreqhdmm

def random_binary_func(shape,seed=2024):
    np.random.seed(seed)

    random_matrix = np.random.choice([-1.0, 1.0], size=(shape, shape))

    return random_matrix

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path='', patience=5, verbose=False, delta=0):
        """
        Args:
            save_path
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        # self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_acc):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

def save_to_binary_file(file_path, byte_data):
    with open(file_path, 'ab' if os.path.exists(file_path) else 'wb') as file:
        file.write(byte_data)

def read_h265_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = file.read()
        file_length = len(file_content)

        return file_content

def numpy_to_bytes(numpy_array):
    if numpy_array.ndim != 1:
        raise ValueError("Input numpy array must be one-dimensional")

    reshaped_array = numpy_array.reshape(-1, 8)
    
    byte_list = []
    
    for row in reshaped_array:
        byte = int(''.join(row.astype(str)), 2)
        byte_list.append(byte)
    
    byte_data = bytes(byte_list)
    
    return byte_data

