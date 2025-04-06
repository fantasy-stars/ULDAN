from skimage.metrics import structural_similarity as ssim

def cal_ssim_tensor(tensor_data_1,tensor_data_2):
    total_ssim=0.0
    for i in range(tensor_data_1.shape[0]):
        img1=tensor_data_1[i].squeeze().detach().cpu().numpy()
        img2=tensor_data_2[i].squeeze().detach().cpu().numpy()

        total_ssim+=ssim(img1,img2,data_range=1)
        
    return total_ssim
