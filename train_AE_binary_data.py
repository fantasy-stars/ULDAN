# from sympy import im
import torch
import torchvision
import numpy as np
import random
import scipy
from torch.utils.data import DataLoader,Dataset,TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from model.net_AE_v1_binary import net_AE_w_tanh_binaryloss_v1
from model.net_AE_v2_binary import net_AE_w_tanh_binaryloss_v2
from model.net_AE_v3_binary import net_AE_w_tanh_binaryloss_v3
from model.mse_binary_loss import mse_binary_loss
from model.mse_binary_orthogonal_loss import mse_binaryorthognal_loss
import argparse
from tqdm import tqdm
from sklearn import preprocessing
from utils.utils import EarlyStopping
from matplotlib import pyplot as plt
from functions.cal_corr_from_tensor import cal_corr_tensor
from functions.cal_ssim_from_tensor import cal_ssim_tensor
from functions.cal_mse_from_tensor import cal_mse_tensor
from functions.cal_acc_from_tensor import cal_acc_tensor
# from skimage.metrics import structural_similarity as ssim
from utils.utils import walsh_func,cake_cutting_func,random_binary_func

import os
import cv2

def main(args):
    print(torch.cuda.is_available())
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_epochs=args.num_epochs
    batch_size=args.batch_size
    learning_rate=args.learning_rate

    shape_x=args.shape_x
    shape_x2=args.shape_x2
    captured_M=args.captured_M

    all_len=args.all_len
    train_len=int(all_len*0.8)
    val_len=int(all_len-train_len)

    dataset_name=args.dataset_name
    all_data_npy_file=args.input_data_npy_file
    ori_imgs_all=np.load(all_data_npy_file)
    print(ori_imgs_all.shape,np.max(ori_imgs_all))

    L1 = [i*5 for i in range(1,val_len)]
    print("L1,",L1[:5])
    X_train,X_test=[],[]
    for i in range(all_len):
        if i not in L1:
            X_train.append(ori_imgs_all[i])
        else:
            X_test.append(ori_imgs_all[i])
    print('dataset len: {}'.format(len(X_train)+len(X_test)),"train_len:",len(X_train),'val_len:',len(X_test))


    x_data=torch.tensor(np.array(X_train).reshape(-1,1,shape_x,shape_x2),dtype=torch.float)
    train_data=TensorDataset(x_data,x_data)
    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=False,drop_last=True)

    x_data=torch.tensor(np.array(X_test).reshape(-1,1,shape_x,shape_x2),dtype=torch.float)
    test_data=TensorDataset(x_data,x_data)
    test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,drop_last=True)

    model_name=args.model_name

    mask_trainable_flag= True if args.mask_mat_is_trainsble else False
    print('mask_trainable_flag, ', mask_trainable_flag)
    # 0-None, 1-HSI, 2-Walsh, 3-CakeCutting, 4-Random_binary
    if args.fixed_mask_mat==4: # Random_binary
        fixed_mask=torch.tensor(1000.0*random_binary_func(shape_x*shape_x2)[:captured_M,:].T,dtype=torch.float).to(device)
    elif args.fixed_mask_mat==3: # CakeCutting
        fixed_mask=torch.tensor(1000.0*cake_cutting_func(shape_x*shape_x2)[:captured_M,:].T,dtype=torch.float).to(device)
    elif args.fixed_mask_mat==2: # Walsh
        fixed_mask=torch.tensor(1000.0*walsh_func(shape_x*shape_x2)[:captured_M,:].T,dtype=torch.float).to(device)
    elif args.fixed_mask_mat==1: # HSI
        # Note: tanh operation is required
        fixed_mask=torch.tensor(1000*scipy.linalg.hadamard(shape_x*shape_x2)[:,:captured_M]).to(device)
    else:
        fixed_mask=None
    
    if model_name=='net_AE_w_tanh_v1':
        model=net_AE_w_tanh_v1(shape_x=shape_x,M=captured_M).to(device)
    elif model_name=='net_AE_w_tanh_binaryloss_v1':
        model=net_AE_w_tanh_binaryloss_v1(shape_x=shape_x,M=captured_M,fixed_mask=fixed_mask).to(device)
        assert args.loss_name in ['mse_binaryorthognal_loss', 'mse_binary_loss', 'mse_binaryv2_loss']
    elif model_name=='net_AE_w_tanh_binaryloss_v2':
        model=net_AE_w_tanh_binaryloss_v2(shape_x=shape_x,M=captured_M,fixed_mask=fixed_mask).to(device)
        assert args.loss_name in ['mse_binaryorthognal_loss', 'mse_binary_loss', 'mse_binaryv2_loss']
    elif model_name=='net_AE_w_tanh_gray2binaryloss_v3':
        model=net_AE_w_tanh_gray2binaryloss_v3(shape_x=shape_x,shape_x2=shape_x2,shape_y=shape_x,M=captured_M,fixed_mask=fixed_mask,mask_trainable=mask_trainable_flag,snr_db=10).to(device)
        assert args.loss_name in ['mse_binaryorthognal_loss', 'mse_binary_loss', 'mse_binaryv2_loss']
    elif model_name=='net_AE_w_tanh_BinaryOrthogonalityloss_v2':
        model=net_AE_w_tanh_BinaryOrthogonalityloss_v2(shape_x=shape_x,M=captured_M).to(device)
        assert args.loss_name in ['mse_binaryorthognal_loss', 'mse_binary_loss', 'mse_binaryv2_loss']
    elif model_name=='net_AE_w_tanh_BinaryOrthogonalityloss_v3':
        model=net_AE_w_tanh_BinaryOrthogonalityloss_v3(shape_x=shape_x,M=captured_M).to(device)
        assert args.loss_name in ['mse_binaryorthognal_loss', 'mse_binary_loss', 'mse_binaryv2_loss']
    else:
        raise NameError('model_name设置不正确')

    loss_name=args.loss_name
    if loss_name=='mse':
        criterion=nn.MSELoss()
    elif loss_name=='mse_binary_loss':
        criterion=mse_binary_loss()
    elif loss_name=='mse_binaryorthognal_loss':
        criterion=mse_binaryorthognal_loss()
    else:
        raise NameError('loss_name设置不正确')
    
    
    if args.fixed_mask_mat==4: # Random_binary
        mask_extra='Random'
    elif args.fixed_mask_mat==3: # CakeCutting
        mask_extra='CakeCutting'
    elif args.fixed_mask_mat==2: # Walsh
        mask_extra='Walsh'
    elif args.fixed_mask_mat==1: # HSI
        mask_extra='HSI'
    else:
        mask_extra='NN'


    saved_model_file=''

    if not os.path.exists(saved_model_file):
        os.makedirs(saved_model_file)
    with open(saved_model_file+'log.txt','a') as f:
        f.writelines([saved_model_file,'\n'])

    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.90,patience=3,verbose=True,min_lr=0.000001)# total_step=len(train_loader)
    
    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    print('Start training...')
    best_ssim,best_corr,best_acc=0.0,0.0,0.0
    best_binary_loss=np.Inf
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch))
        model.train()
        train_loss,train_corr,train_ssim,train_acc=0.0,0.0,0.0,0.0
        train_cnt=0   

        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i,(images,_) in loop:
            images=images.to(device)
            labels=images.to(device)

            if model_name=='net_AE_w_tanh_v1':
                outputs=model(images)
                loss=criterion(outputs,labels)
            elif model_name in ['net_AE_w_tanh_binaryloss_v1', 'net_AE_w_tanh_binaryloss_v2','net_AE_w_tanh_gray2binaryloss_v3','net_AE_w_tanh_BinaryOrthogonalityloss_v2','net_AE_w_tanh_BinaryOrthogonalityloss_v3']:
                outputs, mask_mat=model(images)
                if loss_name in ['mse_binaryv2_loss', 'mse_binary_loss']:
                    loss, mse_loss, binary_loss=criterion(outputs,labels,mask_mat)
                elif loss_name=='mse_binaryorthognal_loss':
                    loss, mse_loss, binary_loss, orthogonality_loss=criterion(outputs,labels,mask_mat)
                else:
                    raise NameError('loss_name设置不正确')
            else:
                raise NameError('model_name设置不正确')


            train_loss+=loss.item()
            train_acc+=cal_acc_tensor(outputs,labels)

            train_cnt+=outputs.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss = train_loss/train_cnt) 
        print('Epoch:{},avg_train_acc:{:.6f}'.format(epoch+1,train_acc/train_cnt))

        model.eval()
        with torch.no_grad():
            val_loss,val_mse,val_corr,val_ssim,val_binary,val_orthogonal,val_acc=0.0,0.0,0.0,0.0,0.0,0.0,0.0
            val_cnt=0   

            for (images,_) in test_loader:
                images = images.to(device)
                labels =images.to(device)

                if model_name=='net_AE_w_tanh_v1':
                    outputs=model(images)
                    loss=criterion(outputs,labels)
                elif model_name in ['net_AE_w_tanh_binaryloss_v1','net_AE_w_tanh_binaryloss_v2', 'net_AE_w_tanh_gray2binaryloss_v3','net_AE_w_tanh_BinaryOrthogonalityloss_v2','net_AE_w_tanh_BinaryOrthogonalityloss_v3']:
                    outputs, mask_mat=model(images)
                    if loss_name in ['mse_binaryv2_loss', 'mse_binary_loss']:
                        loss, mse_loss, binary_loss=criterion(outputs,labels,mask_mat)
                        val_binary+=binary_loss.item()
                    elif loss_name=='mse_binaryorthognal_loss':
                        loss, mse_loss, binary_loss, orthogonality_loss=criterion(outputs,labels,mask_mat)
                        val_orthogonal+=orthogonality_loss.item()
                        val_binary+=binary_loss.item()
                    else:
                        raise NameError('loss_name设置不正确')
                else:
                    raise NameError('model_name设置不正确')

                val_loss+=loss.item()
                val_corr+=cal_corr_tensor(outputs,labels)
                val_ssim+=cal_ssim_tensor(outputs,labels)
                val_mse+=cal_mse_tensor(outputs,labels)
                val_acc+=cal_acc_tensor(outputs,labels)

                val_cnt+=outputs.shape[0]

            print("Epoch:{},avg_eval_loss:{:.6f},avg_eval_acc:{:.6f}, binary_loss:{:.6f}".format(epoch+1,val_loss/val_cnt,val_acc/val_cnt,val_binary/val_cnt))
        
        if 'binary' in loss_name and 'orthognal' not in loss_name:
            with open(saved_model_file+'log.txt','a') as f:
                f.writelines(['Epoch:{}, avg_train_loss:{:.6f}, avg_eval_loss:{:.6f}, avg_eval_mse:{:.6f}, avg_eval_corr:{:.4f}, avg_eval_ssim:{:.4f}, avg_eval_binary:{:.6f}, avg_eval_acc:{:.4f}'.format(epoch+1,train_loss/train_cnt,val_loss/val_cnt,val_mse/val_cnt,val_corr/val_cnt,val_ssim/val_cnt,val_binary/val_cnt,val_acc/val_cnt),'\n'])
        elif 'binary' in loss_name and 'orthognal' in loss_name:
            with open(saved_model_file+'log.txt','a') as f:
                f.writelines(['Epoch:{}, avg_train_loss:{:.6f}, avg_eval_loss:{:.6f}, avg_eval_mse:{:.6f}, avg_eval_corr:{:.4f}, avg_eval_ssim:{:.4f}, avg_eval_binary:{:.6f}, avg_eval_orthognal:{:.4f}, avg_eval_acc:{:.4f}'.format(epoch+1,train_loss/train_cnt,val_loss/val_cnt,val_mse/val_cnt,val_corr/val_cnt,val_ssim/val_cnt,val_binary/val_cnt,val_orthogonal/val_cnt,val_acc/val_cnt),'\n'])
        else:
            with open(saved_model_file+'log.txt','a') as f:
                f.writelines(['Epoch:{}, avg_train_loss:{:.6f}, avg_eval_loss:{:.6f}, avg_eval_mse:{:.6f}, avg_eval_corr:{:.4f}, avg_eval_ssim:{:.4f}, avg_eval_acc:{:.4f}'.format(epoch+1,train_loss/train_cnt,val_loss/val_cnt,val_mse/val_cnt,val_corr/val_cnt,val_ssim/val_cnt,val_acc/val_cnt),'\n'])


        if (val_acc/val_cnt > best_acc) or (best_binary_loss>val_binary/val_cnt):
            best_acc=max(val_acc/val_cnt,best_acc)
            best_binary_loss=min(best_binary_loss,val_binary/val_cnt)
            torch.save(model.state_dict(),'{}epoch_{:0>3d}_len_{}_eval_corr_{:.4f}_eval_acc_{:.4f}.pth'.format(saved_model_file,epoch,all_len,val_corr/val_cnt,val_acc/val_cnt))
        lr_scheduler.step(val_loss)

        early_stopping(-best_binary_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=0.001)    
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--captured_M", type=int, default=0)
    parser.add_argument("--shape_x", type=int, default=64)
    parser.add_argument("--shape_x2", type=int, default=64)
    parser.add_argument("--expected_ssim", type=float, default=0.6)   
    parser.add_argument("--early_stop_patience", type=int, default=200)     

    parser.add_argument("--fixed_mask_mat", type=int, default=0,help='0-None, 1-HSI, 2-Walsh, 3-CakeCutting, 4-Random_binary')  
    parser.add_argument("--mask_mat_is_trainsble", type=bool, default=1)  
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--loss_name", type=str, default="mse_binary_loss")

    parser.add_argument("--all_len", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--input_data_npy_file", type=str, default='')


    args = parser.parse_args()

    main(args)


    print('finsh')
