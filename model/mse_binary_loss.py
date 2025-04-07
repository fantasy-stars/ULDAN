import torch
import torch.nn as nn

class mse_binary_loss(nn.Module):
    def __init__(self):
        super(mse_binary_loss,self).__init__()
    
    def forward(self,x,y,mask_mat):
        x=torch.squeeze(x)
        y=torch.squeeze(y)

        mse_loss=torch.mean((x-y)**2,dim=[0,1,2])

        binary_loss=torch.mean(((1-torch.tanh(mask_mat))**2)*((1+torch.tanh(mask_mat))**2),dim=[0,1])

        combined_loss=mse_loss+binary_loss

        return combined_loss, mse_loss, binary_loss


if __name__ == "__main__":
    N=20
    x=torch.rand((3,1,N,N))
    y=torch.rand((3,1,N,N))

    mask_mat=torch.rand((786,66))

    my_loss=mse_binary_loss()
    combined_loss, mse_loss, binary_loss=my_loss(x,y,mask_mat)

    print('combined_loss, mse_loss, binary_loss, ',combined_loss, mse_loss, binary_loss)

        