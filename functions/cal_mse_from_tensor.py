import numpy as np

def cal_mse_tensor(tensor_data_1,tensor_data_2):
    total_mse=0.0
    for i in range(tensor_data_1.shape[0]):
        img1=tensor_data_1[i].squeeze().detach().cpu().numpy()
        img2=tensor_data_2[i].squeeze().detach().cpu().numpy()

        total_mse+=np.sum((img1-img2)**2)/(img1.size)
        
    return total_mse
