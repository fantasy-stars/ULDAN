import torch
import torch.nn as nn

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class OrthogonalityLoss(nn.Module):
    def __init__(self):
        super(OrthogonalityLoss, self).__init__()

    def forward(self, weight_matrix):
        weight_matrix_transpose = weight_matrix.t()
        dot_products = torch.matmul(weight_matrix_transpose, weight_matrix) # M*M
        lines_dots=torch.tensor(weight_matrix.size()[0]).to(device)
        eye_mat=torch.eye(weight_matrix.size()[1]).to(device)
        orthogonality_loss = torch.sum((dot_products-lines_dots*eye_mat)**2)/ (weight_matrix.numel()-weight_matrix.size()[1]) / lines_dots**4
        
        return orthogonality_loss

class mse_binaryorthognal_loss(nn.Module):
    def __init__(self):
        super(mse_binaryorthognal_loss,self).__init__()
        self.OrthogonalityLoss=OrthogonalityLoss()
    
    def forward(self,x,y,mask_mat):
        x=torch.squeeze(x)
        y=torch.squeeze(y)

        mse_loss=torch.mean((x-y)**2,dim=[0,1,2])

        binary_loss=torch.mean(((1-torch.tanh(mask_mat))**2)*((1+torch.tanh(mask_mat))**2),dim=[0,1])

        orthogonality_loss=self.OrthogonalityLoss(torch.tanh(mask_mat))/100
        
        combined_loss=mse_loss + binary_loss + orthogonality_loss

        return combined_loss, mse_loss, binary_loss, orthogonality_loss



if __name__ == "__main__":
    N=20
    x=torch.rand((3,1,N,N))
    y=torch.rand((3,1,N,N))

    mask_mat=torch.rand((786,66))

    my_loss=mse_binaryorthognal_loss()
    combined_loss, mse_loss, binary_loss, orthogonality_loss=my_loss(x,y,mask_mat)

    print('combined_loss, mse_loss, binary_loss, orthogonality_loss, ',combined_loss, mse_loss, binary_loss, orthogonality_loss)

        