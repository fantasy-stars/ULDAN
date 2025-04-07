import numpy as np

def cal_acc_tensor(tensor_data_1,tensor_data_2):
    total_acc=0.0
    data1=tensor_data_1.squeeze().detach().cpu().numpy()
    data2=tensor_data_2.squeeze().detach().cpu().numpy()

    data1=np.where(data1 >= 0.5, 1, 0)
    data2=np.where(data2 >= 0.5, 1, 0)

    total_acc=np.sum(data1==data2)/data1.size*data1.shape[0]

    return total_acc

