from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
from models4DIP import *
import torch
import torch.optim
# from skimage.measure import compare_psnr
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from models4DIP.denoising_utils import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

#------------------------------------------------------------------------------------#
# Deep Image Prior, CVPR 2018, Dmitry Ulyanov, Andrea Vedaldi, Victor Lempitsky
# https://github.com/DmitryUlyanov/deep-image-prior
#------------------------------------------------------------------------------------#

imsize = -1
PLOT =  True
sigma = 25
sigma_ = sigma / 255.


## denoising
fname_flag = 'denoising'

# gt file
fname = ''
fname_noisy = ''
saved_root_file=''
os.makedirs(saved_root_file,exist_ok=True)
psnr_list_all=[]
#----------------------------------------------------------------------------------------

if fname_flag == 'denoising':

    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    # img_pil = crop_image(get_image(fname, imsize)[0].convert('L'), d=32)
    img_np = pil_to_np(img_pil)


    img_noisy_pil = crop_image(get_image(fname_noisy, imsize)[0].convert('RGB'), d=32)
    # img_noisy_pil = crop_image(get_image(fname_noisy, imsize)[0].convert('L'), d=32)
    img_noisy_np = pil_to_np(img_noisy_pil)

    print(img_np.shape, img_noisy_np.shape)
    print(np.max(img_np))

    # if PLOT:
    #     plot_image_grid([img_np, img_noisy_np], 4, 6); cv2.cvtColor(np.transpose((img_np*255).astype(np.int8), (1, 2, 0)), cv2.COLOR_RGB2BGR)

else:
    assert False

INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 0.01

OPTIMIZER = 'adam'  # 'LBFGS'
show_every = 100
exp_weight = 0.99

if fname_flag == 'denoising':
    num_iter = 3000
    input_depth = 32
    figsize = 4

    net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128,
                    skip_n33u=128,
                    skip_n11=4,
                    num_scales=5,
                    upsample_mode='bilinear').type(dtype)

else:
    assert False

net_input = get_noise(input_depth, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()

# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()]);
print('Number of params: %d' % s)

# Loss
mse = torch.nn.MSELoss().type(dtype)

img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)

net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0

def closure():
    global i, out_avg, psrn_noisy_last, last_net, net_input

    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)

    out = net(net_input)

    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = mse(out, img_noisy_torch)
    total_loss.backward()

    psrn_noisy = compare_psnr(img_noisy_np, out.detach().cpu().numpy()[0])
    psrn_gt = compare_psnr(img_np, out.detach().cpu().numpy()[0])
    psrn_gt_sm = compare_psnr(img_np, out_avg.detach().cpu().numpy()[0])


    print('Iteration %05d    Loss %f   PSNR_noisy: %f   PSRN_gt: %f PSNR_gt_sm: %f' % (
    i, total_loss.item(), psrn_noisy, psrn_gt, psrn_gt_sm), '\r', end='')

    # Backtracking
    if i % show_every:
        if psrn_noisy - psrn_noisy_last < -5:
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return total_loss * 0
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psrn_noisy_last = psrn_noisy

    i += 1

    return total_loss


p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)

np.save(os.path.join(saved_root_file,'psnr_all.npy'),np.array(psnr_list_all))

