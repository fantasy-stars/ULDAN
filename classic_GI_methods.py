import cv2
import os
import numpy as np
from tqdm import tqdm
import argparse

def mkdir_func(path):
    if not os.path.exists(path):
        os.makedirs(path)

def GI_func(src,iter_cnt=1000,luminance_index=20):
    w,h=src.shape[0], src.shape[1] 

    In_sum, Bn_sum, Bn_In_sum=0, 0, 0
    for _ in tqdm(range(iter_cnt)):
        D=np.random.random((w,h))
        E=src*D

        Bn_sum+=np.mean(E)
        In_sum+=D
        Bn_In_sum+=np.mean(E)*D

    print('reconstruct original image...')
    # G=np.mean(Bn_In)-np.mean(Bn)*np.mean(In)
    G=Bn_In_sum/iter_cnt-(Bn_sum/iter_cnt)*(In_sum/iter_cnt)
    G*=luminance_index
    G=255*((G-np.min(G))/(np.max(G)-np.min(G)))

    return G

def DGI_func(src,iter_cnt=1000,luminance_index=20):
    w,h=src.shape[0], src.shape[1] 

    Bn_sum, Rn_sum, Bn_In_sum, Rn_In_sum=0,0,0,0
    for _ in tqdm(range(iter_cnt)):
        D=np.random.random((w,h))
        E=src*D

        Bn_sum+=np.mean(E)
        Rn_sum+=np.mean(D)
        Bn_In_sum+=np.mean(E)*D
        Rn_In_sum+=np.mean(D)*D

    print('reconstruct original image...')
    G=Bn_In_sum/iter_cnt-(Bn_sum/iter_cnt)/(Rn_sum/iter_cnt)*(Rn_In_sum/iter_cnt)

    G*=luminance_index

    G=255*((G-np.min(G))/(np.max(G)-np.min(G)))

    return G

def NGI_func(src,iter_cnt=1000,luminance_index=20):
    w,h=src.shape[0], src.shape[1] 

    Bn_sum, Rn_sum, In_sum, Bn_In_div_Rn_sum = 0,0,0,0
    for _ in tqdm(range(iter_cnt)):
        D=np.random.random((w,h))
        E=src*D

        Bn_sum+=np.mean(E)
        In_sum+=D
        Rn_sum+=np.mean(D)
        Bn_In_div_Rn_sum+=(np.mean(E)/np.mean(D))*D

    print('reconstruct original image...')
    G=(Bn_In_div_Rn_sum/iter_cnt)-(Bn_sum/iter_cnt)/(Rn_sum/iter_cnt)*(In_sum/iter_cnt)

    G*=luminance_index

    G=255*((G-np.min(G))/(np.max(G)-np.min(G)))

    return G

def main(GI_cnt=0,demo_file='demo_img/Lena',new_size=64,iter_cnt=1000,luminance_index=20,saved_flag=False, display_flag=False):
    src=cv2.imread(os.path.join(demo_file,'original.png'),cv2.IMREAD_GRAYSCALE)

    src=src/255.0

    src=cv2.resize(src,(new_size,new_size))

    if GI_cnt==0:
        extra='GI'
        G=GI_func(src,iter_cnt,luminance_index)
    elif GI_cnt==1:
        extra='DGI'
        G=DGI_func(src,iter_cnt,luminance_index)
    elif GI_cnt==2:
        extra='NGI'
        G=NGI_func(src,iter_cnt,luminance_index)

    if saved_flag:
        print('save reconstructed image...')
        saved_file=os.path.join(demo_file,'{}_size{}'.format(extra,new_size))
        mkdir_func(saved_file)
        cv2.imwrite(os.path.join(saved_file,'{}_size_{}_iter_{:0>6d}_index_{}.png'.format(extra,new_size,iter_cnt,luminance_index)),np.array(G).astype(np.uint8))
    if display_flag:
        print('display reconstructed image...')
        cv2.imshow('{}'.format(extra),np.array(G).astype(np.uint8))
        cv2.waitKey(0)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--GI_cnt", type=int, default=2, help='0-GI  1-DGI  2-NGI')
    parser.add_argument("--demo_file", type=str, default='demo_img/Lena')
    parser.add_argument("--new_size", type=int, default=64)
    parser.add_argument("--iter_cnt", type=int, default=64*64)
    parser.add_argument("--luminance_index", type=int, default=20)
    parser.add_argument("--saved_flag", type=bool, default=1,help='Whether to save the reconstructed image')
    parser.add_argument("--display_flag", type=bool, default=0,help='Whether to display the reconstructed image')

    args = parser.parse_args()

    main(args.GI_cnt,args.demo_file,args.new_size,args.iter_cnt,args.luminance_index,args.saved_flag, args.display_flag)

