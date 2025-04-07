import torch
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
from model.target_net_decoder_v1 import target_net_decoder_v1
from model.target_net_decoder_v2 import target_net_decoder_v2
from utils.utils import EarlyStopping
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt
from functions.cal_corr_from_tensor import cal_corr_tensor
from functions.cal_ssim_from_tensor import cal_ssim_tensor
from functions.cal_mse_from_tensor import cal_mse_tensor
from functions.cal_acc_from_tensor import cal_acc_tensor
from utils.utils import set_requires_grad
import os

def main(args):

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_epochs=args.num_epochs
    batch_size=args.batch_size
    learning_rate=args.learning_rate

    shape_y=args.shape_y
    captured_M=args.captured_M

    all_len=args.all_len
    train_len=int(all_len*0.8)
    val_len=int(all_len-train_len)

    dataset_name=args.dataset_name
    input_X_data_npy_file=args.input_X_data_npy_file
    assert args.mask_mat_type_list[args.mask_mat_type] in input_X_data_npy_file
    input_Y_data_npy_file=args.input_Y_data_npy_file

    X_data_all=np.load(input_X_data_npy_file)
    Y_data_all=np.load(input_Y_data_npy_file)

    L1 = [i*5 for i in range(val_len)]
    X_train,Y_train=[],[]
    X_test,Y_test=[],[]
    for i in range(all_len):
        if i not in L1:
            X_train.append(X_data_all[i])
            Y_train.append(Y_data_all[i])
        else:
            X_test.append(X_data_all[i])
            Y_test.append(Y_data_all[i])
    print('dataset len: {}'.format(len(X_train)+len(X_test)),"train_len:",len(X_train),'val_len:',len(X_test))

    x_data=torch.tensor(np.array(X_train).reshape(-1,captured_M),dtype=torch.float)
    y_data=torch.tensor(np.array(Y_train).reshape(-1,1,shape_y,shape_y),dtype=torch.float)
    train_data=TensorDataset(x_data,y_data)
    train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=False,drop_last=True)

    x_data=torch.tensor(np.array(X_test).reshape(-1,captured_M),dtype=torch.float)
    test_data=TensorDataset(x_data,y_data)
    test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False,drop_last=True)

    model_name=args.model_name

    if model_name=='target_net_decoder_v1':
        model=target_net_decoder_v1(M=captured_M,shape_y=shape_y).to(device)
    elif model_name=='target_net_decoder_v2':
        model=target_net_decoder_v2(M=captured_M,shape_y=shape_y).to(device)
    else:
        raise NameError('Wrong: model_name.....')

    if args.fix_reconstruct:
        print('fix_reconstruct...')
        model.load_state_dict(torch.load(args.pretrain_pth_name))

    loss_name=args.loss_name
    if loss_name=='mse':
        criterion=nn.MSELoss()
    else:
        raise NameError('Wrong: loss_name.....')

    saved_model_file=args.saved_log_root_file
    os.makedirs(saved_model_file,exist_ok=True)
    with open(os.path.join(saved_model_file, 'log.txt'),'a') as f:
        f.writelines([saved_model_file,'\n'])

    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',factor=0.90,patience=3,verbose=True,min_lr=0.000001)
    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    print('Start training...')
    best_ssim,best_corr,best_acc=0.0,0.0,0.0
    for epoch in range(num_epochs):
        print("Epoch:{}".format(epoch))
        model.train()
        if args.fix_reconstruct:
            set_requires_grad(model.reconstruct, requires_grad=False)
        train_loss,train_corr,train_ssim=0.0,0.0,0.0
        train_cnt=0

        loop = tqdm(enumerate(train_loader), total =len(train_loader))
        for i,(images,labels) in loop:
            images=images.to(device)
            labels=labels.to(device)

            if model_name in ['target_net_decoder_v1', 'target_net_decoder_v2']:
                outputs=model(images)
                loss=criterion(outputs,labels)
            else:
                raise NameError('Wrong: model_name.....')

            train_loss+=loss.item()

            train_cnt+=outputs.shape[0]
            # train_corr+=cal_corr_tensor(outputs,labels)
            # train_ssim+=cal_ssim_tensor(outputs,labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss = train_loss/train_cnt)

        model.eval()
        with torch.no_grad():
            val_loss,val_mse,val_corr,val_ssim,val_acc=0.0,0.0,0.0,0.0,0.0
            val_cnt=0

            for (images,labels) in test_loader:
                images = images.to(device)
                labels =labels.to(device)

                if model_name in ['target_net_decoder_v1', 'target_net_decoder_v2']:
                    outputs=model(images)
                    loss=criterion(outputs,labels)
                else:
                    raise NameError('Wrong: model_name.....')

                val_loss+=loss.item()
                val_corr+=cal_corr_tensor(outputs,labels)
                val_ssim+=cal_ssim_tensor(outputs,labels)
                val_mse+=cal_mse_tensor(outputs,labels)
                if not args.gray2binary_flag:
                    val_acc+=cal_acc_tensor(outputs,labels)

                val_cnt+=outputs.shape[0]

            if not args.gray2binary_flag:
                print("Epoch:{},avg_eval_loss:{:.6f},avg_eval_acc:{:.6f}".format(epoch+1,val_loss/val_cnt,val_acc/val_cnt))
            else:
                print("Epoch:{},avg_eval_corr:{:.6f},avg_eval_ssim:{:.6f}".format(epoch+1,val_corr/val_cnt,val_ssim/val_cnt))


        with open(os.path.join(saved_model_file, 'log.txt'),'a') as f:
            f.writelines(['Epoch:{}, avg_train_loss:{:.6f}, avg_eval_loss:{:.6f}, avg_eval_mse:{:.6f}, avg_eval_corr:{:.4f}, avg_eval_ssim:{:.4f}, avg_eval_acc:{:.4f}'.format(epoch+1,train_loss/train_cnt,val_loss/val_cnt,val_mse/val_cnt,val_corr/val_cnt,val_ssim/val_cnt,val_acc/val_cnt),'\n'])

        if not args.gray2binary_flag:
            if val_acc/val_cnt > best_acc:
                best_acc=val_acc/val_cnt
                torch.save(model.state_dict(),os.path.join(saved_model_file,'epoch_{:0>3d}_eval_corr_{:.4f}_eval_acc_{:.4f}.pth'.format(epoch,val_corr/val_cnt,val_acc/val_cnt)))
        else:
            if (val_corr/val_cnt > best_corr) or (val_ssim/val_cnt > best_ssim):
                best_corr=max(best_corr,val_corr/val_cnt)
                best_ssim=max(best_ssim,val_ssim/val_cnt)
                torch.save(model.state_dict(),os.path.join(saved_model_file,'epoch_{:0>3d}_eval_corr_{:.4f}_eval_ssim_{:.4f}.pth'.format(epoch,val_corr/val_cnt,val_ssim/val_cnt)))

        lr_scheduler.step(val_loss)

        if not args.gray2binary_flag:
            early_stopping(best_acc)
        else:
            early_stopping(best_ssim+best_corr)

        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--early_stop_patience", type=int, default=15)

    parser.add_argument("--captured_M", type=int, default=1638)
    parser.add_argument("--shape_y", type=int, default=32)
    parser.add_argument("--expected_ssim", type=float, default=0.6)

    # parser.add_argument("--model_name", type=str, default="target_net_decoder_v1")
    parser.add_argument("--model_name", type=str, default="target_net_decoder_v2")
    parser.add_argument("--loss_name", type=str, default="mse")

    parser.add_argument("--all_len", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default='')
    parser.add_argument("--gray2binary_flag", type=int, default=1)
    parser.add_argument("--saved_log_root_file", type=str, default='')
    parser.add_argument("--input_X_data_npy_file", type=str, default='')
    parser.add_argument("--input_Y_data_npy_file", type=str, default='')
    parser.add_argument("--fix_reconstruct", type=bool, default=1)
    parser.add_argument("--pretrain_pth_name", type=str, default='')

    parser.add_argument("--mask_mat_type", type=int, default=0,help='0-NN, 1-HSI')
    parser.add_argument("--mask_mat_type_list", type=list, default=['NN','HSI'])

    args = parser.parse_args()

    main(args)


