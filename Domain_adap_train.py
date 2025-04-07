import torch
import os
from torch import nn
from utils.utils import set_requires_grad,loop_iterable,test_func_acc,test_func_corr_ssim,split_train_test
from utils.utils import EarlyStopping
from model.D_net_v1 import discriminator_v1
from model.target_net_decoder_v1 import target_net_decoder_v1
from model.target_net_decoder_v2 import target_net_decoder_v2
from model.source_net_decoder_v1 import source_net_decoder_v1
from model.source_net_decoder_v2 import source_net_decoder_v2
import numpy as np
from tqdm import tqdm
import argparse

def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(2023)
    torch.cuda.manual_seed_all(2023)
    np.random.seed(2023)

    shape_y=args.shape_y
    captured_M=args.captured_M
    all_len=args.all_len

    X_source=np.load(os.path.join(args.input_source_data_file))
    Y_source=np.load(args.input_Y_data_npy_file)[:all_len]
    X_target=np.load(os.path.join(args.input_target_data_file))

    all_len=args.all_len
    train_len=int(all_len*0.8)
    val_len=int(all_len-train_len)
    
    train_source_loader, test_source_loader=split_train_test(all_len,val_len,X_source,Y_source,captured_M,shape_y,args.batch_size)
    train_target_loader, test_target_loader=split_train_test(all_len,val_len,X_target,Y_source,captured_M,shape_y,args.batch_size)

    source_model = source_net_decoder_v2(M=captured_M,shape_y=shape_y).to(device)
    source_model.load_state_dict(torch.load(args.source_model_pth))
    source_model.eval()
    set_requires_grad(source_model, requires_grad=False)

    clf = source_model
    source_model = source_model.feature

    target_model = target_net_decoder_v2(M=captured_M,shape_y=shape_y).to(device)
    target_model.load_state_dict(torch.load(args.source_model_pth))
    target_model = target_model.feature

    discriminator=discriminator_v1(shape_y=shape_y).to(device)

    discriminator_optim = torch.optim.Adam(discriminator.parameters(),lr=args.disc_lr)
    target_optim = torch.optim.Adam(target_model.parameters(),lr=args.clf_lr)
    criterion = nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=args.early_stop_patience)

    saved_log_file=args.saved_log_root_file
    os.makedirs(saved_log_file,exist_ok=True)
    with open(os.path.join(saved_log_file,'log.txt'),'a') as f:
        f.writelines([saved_log_file,'\n','source_model_pth:{}'.format(args.source_model_pth),'\n','batchsize:{}, iters:{}, epochs:{}, k-disc:{}, k-clf:{}, disc_lr:{}, clf_lr:{}'.format(args.batch_size,args.iterations,args.epochs,args.k_disc,args.k_clf,args.disc_lr,args.clf_lr),'\n'])

    best_avg_predict_target_acc=0.8
    best_avg_predict_target_corr,best_avg_predict_target_ssim=0.0,0.0
    for epoch in range(1, args.epochs+1):
        batch_iterator = zip(loop_iterable(train_source_loader), loop_iterable(train_target_loader))

        print('epoch, ',epoch)

        total_loss = 0
        total_accuracy = 0
        for _ in tqdm(range(args.iterations)):
            # Train discriminator
            set_requires_grad(target_model, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)
            for _ in range(args.k_disc):
                (source_x, _), (target_x, _) = next(batch_iterator)
                source_x, target_x = source_x.to(device), target_x.to(device)

                source_features = source_model(source_x).view(source_x.shape[0], -1)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                discriminator_x = torch.cat([source_features, target_features])
                discriminator_y = torch.cat([torch.ones(source_x.shape[0], device=device),
                                                torch.zeros(target_x.shape[0], device=device)])

                preds = discriminator(discriminator_x).squeeze()
                loss = criterion(preds, discriminator_y)

                discriminator_optim.zero_grad()
                loss.backward()
                discriminator_optim.step()

                total_loss += loss.item()
                total_accuracy += ((preds > 0).long() == discriminator_y.long()).float().mean().item()

            # Train target_model
            set_requires_grad(target_model, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)
            for _ in range(args.k_clf):
                _, (target_x, _) = next(batch_iterator)
                target_x = target_x.to(device)
                target_features = target_model(target_x).view(target_x.shape[0], -1)

                # flipped labels
                discriminator_y = torch.ones(target_x.shape[0], device=device)

                preds = discriminator(target_features).squeeze()
                loss = criterion(preds, discriminator_y)

                target_optim.zero_grad()
                loss.backward()
                target_optim.step()

            mean_loss = total_loss / (args.iterations*args.k_disc)
            mean_accuracy = total_accuracy / (args.iterations*args.k_disc)

            clf.feature = target_model

            if args.Y_data_is_binary:
                avg_predict_source_acc=test_func_acc(clf,test_source_loader,device)
                avg_predict_target_acc=test_func_acc(clf,test_target_loader,device)
    
                if avg_predict_target_acc>best_avg_predict_target_acc:
                    best_avg_predict_target_acc=avg_predict_target_acc
                    torch.save(clf.state_dict(),os.path.join(saved_log_file, "epoch_{}_Acc_dis_{:.4f}_sour_{:.4f}_targ_{:.4f}.pth".format(epoch,mean_accuracy,avg_predict_source_acc,avg_predict_target_acc)))
            
                with open(os.path.join(saved_log_file,'log.txt'),'a') as f:
                        f.writelines(['Epoch:{}, discriminator_loss:{:.4f}, discriminator_acc:{:.4f}, source_acc:{:.4f}, target_acc:{:.4f}'.format(epoch,mean_loss,mean_accuracy,avg_predict_source_acc,avg_predict_target_acc),'\n'])
            else:
                avg_predict_source_corr,avg_predict_source_ssim=test_func_corr_ssim(clf,test_source_loader,device)
                avg_predict_target_corr,avg_predict_target_ssim=test_func_corr_ssim(clf,test_target_loader,device)
    
                if avg_predict_target_corr>best_avg_predict_target_corr or avg_predict_target_ssim>best_avg_predict_target_ssim:
                    best_avg_predict_target_corr=max(best_avg_predict_target_corr,avg_predict_target_corr)
                    best_avg_predict_target_ssim=max(best_avg_predict_target_ssim,avg_predict_target_ssim)
                    torch.save(clf.state_dict(),os.path.join(saved_log_file, "epoch_{}_Acc_dis_{:.4f}_sour_{:.4f}_{:.4f}_targ_{:.4f}_{:.4f}.pth".format(epoch,mean_accuracy,avg_predict_source_corr,avg_predict_source_ssim,avg_predict_target_corr,avg_predict_target_ssim)))
            
                with open(os.path.join(saved_log_file,'log.txt'),'a') as f:
                        f.writelines(['Epoch:{}, discriminator_loss:{:.4f}, discriminator_acc:{:.4f}, source_corr:{:.4f}, source_ssim:{:.4f}, target_corr:{:.4f}, target_ssim:{:.4f}'.format(epoch,mean_loss,mean_accuracy,avg_predict_source_corr,avg_predict_source_ssim,avg_predict_target_corr,avg_predict_target_ssim),'\n'])
            

        if args.Y_data_is_binary:
            print('Epoch:{}, discriminator_loss:{:.4f}, discriminator_acc:{:.4f}, source_acc:{:.4f}, target_acc:{:.4f}'.format(epoch,mean_loss,mean_accuracy,avg_predict_source_acc,avg_predict_target_acc))
            early_stopping(avg_predict_target_acc)
        else:
            early_stopping(best_avg_predict_target_corr+best_avg_predict_target_ssim)
            print('Epoch:{}, discriminator_loss:{:.4f}, discriminator_acc:{:.4f}, source_corr:{:.4f}, source_ssim:{:.4f}, target_corr:{:.4f}, target_ssim:{:.4f}'.format(epoch,mean_loss,mean_accuracy,avg_predict_source_corr,avg_predict_source_ssim,avg_predict_target_corr,avg_predict_target_ssim))
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument("--captured_M", type=int, default=1638)
    arg_parser.add_argument("--shape_y", type=int, default=32)
    arg_parser.add_argument("--early_stop_patience", type=int, default=2)    
    
    arg_parser.add_argument("--all_len", type=int, default=20000)
    arg_parser.add_argument("--input_source_data_file", type=str, default='')
    arg_parser.add_argument("--saved_log_root_file", type=str, default='')
    arg_parser.add_argument("--input_target_data_file", type=str, default='')
    arg_parser.add_argument("--dataset_name", type=str, default='')
    arg_parser.add_argument("--Y_data_is_binary", type=bool, default=0)    
    arg_parser.add_argument("--input_Y_data_npy_file", type=str, default='')
    arg_parser.add_argument("--source_model_pth", type=str, default='')

    arg_parser.add_argument("--mask_mat_type", type=int, default=0,help='0-NN, 1-HSI')  
    arg_parser.add_argument("--mask_mat_type_list", type=list, default=['NN','HSI'])  
   
    arg_parser.add_argument('--batch_size', type=int, default=128)
    arg_parser.add_argument('--iterations', type=int, default=2)
    arg_parser.add_argument('--epochs', type=int, default=2) # run iterations times for each epoch

    arg_parser.add_argument('--k-disc', type=int, default=2) # discriminator iterations in an iteration
    arg_parser.add_argument('--k-clf', type=int, default=40) # feature_extractor iterations in an iteration

    arg_parser.add_argument('--disc_lr', type=float, default=0.001,help='learning rate of discriminator')
    arg_parser.add_argument('--clf_lr', type=float, default=0.001,help='learning rate of target feature_extractor')
    args = arg_parser.parse_args()

    main(args)

