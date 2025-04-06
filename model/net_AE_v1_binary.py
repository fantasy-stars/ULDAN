import torch.nn as nn
import torch
import torch
import torch.nn as nn

class net_AE_w_tanh_binaryloss_v1(nn.Module):
    def __init__(self, shape_x=128, M=666, fixed_mask=None):
        super(net_AE_w_tanh_binaryloss_v1, self).__init__()
        self.shape_x = shape_x
        
        if fixed_mask is None:
            self.mask_mat = nn.Parameter(torch.randn(shape_x**2, M))
        else:
            self.mask_mat = torch.tensor(fixed_mask, requires_grad=False)

        self.feature = nn.Sequential(
            nn.Linear(M, shape_x**2),
            nn.ReLU(),
            nn.BatchNorm1d(shape_x**2),  
        )
        self.reconstruct = nn.Sequential(
            # conv1
            nn.Conv2d(1, 64, 9, 1, 4),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            # conv2
            nn.Conv2d(64, 32, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),
            # conv3
            nn.Conv2d(32, 1, 5, 1, 2),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.matmul(x, self.tanh(self.mask_mat))
        x = self.feature(x)
        x = x.view(-1, 1, self.shape_x, self.shape_x)
        x = self.reconstruct(x)

        return x, self.mask_mat


if __name__ == "__main__":
    shape_x=32
    M=66

    network = net_AE_w_tanh_binaryloss_v1(shape_x=shape_x,M=M)
    input_tensor = torch.randn(10,shape_x,shape_x)
    output_tensor, out_maskmat = network(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
    print("Output mask mat shape:", out_maskmat.shape)