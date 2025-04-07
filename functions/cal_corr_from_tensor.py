import numpy as np

def cal_corr_tensor(tensor_data_1,tensor_data_2):
    total_corr=0.0
    for i in range(tensor_data_1.shape[0]):
        img1=tensor_data_1[i].squeeze().detach().cpu().numpy()
        img2=tensor_data_2[i].squeeze().detach().cpu().numpy()

        img1= img1.reshape(img1.size, order='C')
        img2= img2.reshape(img2.size, order='C')

        total_corr+=np.corrcoef(img1, img2)[0, 1]
        
    return total_corr

