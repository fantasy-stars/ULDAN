from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2
import math
import os

def ssim_func(img1,img2):
    tmp=ssim(img1,img2,data_range=1)
    tmp=max(tmp,0)

    return tmp

def psnr_func(img1,img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def corr_func(img1,img2):
    img1= img1.reshape(img1.size, order='C')
    img2= img2.reshape(img2.size, order='C')
    
    return np.corrcoef(img1, img2)[0, 1]

def acc_func(img1,img2):
    img1=np.where(img1>0.5, 1.0, 0.0)
    img2=np.where(img2>0.5, 1.0, 0.0)

    acc=np.sum(img1==img2)/img1.size

    return acc
