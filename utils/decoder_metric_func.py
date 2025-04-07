from skimage.metrics import structural_similarity as ssim
import numpy as np
import math

def ssim_func(img1,img2):
    if np.mean(img1)<1e-4:
        return 0

    tmp=ssim(img1,img2,data_range=1)
    tmp=max(tmp,0)

    return tmp

def psnr_func(img1,img2):
    if np.mean(img1)<1e-4:
        return 0

    mse = np.mean( (img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def corr_func(img1,img2):
    if np.mean(img1)<1e-4:
        return 0

    img1= img1.reshape(img1.size, order='C')
    img2= img2.reshape(img2.size, order='C')
    
    return np.corrcoef(img1, img2)[0, 1]

def decoder_bit2gray(src):
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

def cal_decoder_matrics(outputs_tensor,labels_target_tensor):
    val_de_mse,val_de_corr,val_de_ssim,val_de_psnr=0.0,0.0,0.0,0.0

    outputs=np.squeeze(outputs_tensor.detach().cpu().numpy())
    outputs=np.where(outputs>0.5,1,0)
    labels_target=labels_target_tensor.detach().cpu().numpy()

    recover_imgs=np.zeros_like(labels_target,dtype=np.float64)
    for cnt in range(recover_imgs.shape[0]):
        recover_imgs[cnt]=decoder_bit2gray(outputs[cnt])/255.0
        val_de_mse+=np.mean((recover_imgs[cnt]-labels_target[cnt])**2)
        val_de_corr+=corr_func(recover_imgs[cnt], labels_target[cnt])
        val_de_ssim+=ssim_func(recover_imgs[cnt], labels_target[cnt])
        val_de_psnr+=psnr_func(recover_imgs[cnt], labels_target[cnt])

    return val_de_mse,val_de_corr,val_de_ssim,val_de_psnr

