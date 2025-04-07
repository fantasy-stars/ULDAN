import torch.nn as nn
import torch
import torch
import torch.nn as nn

# Can be replaced with the desired noise dist.
class NoisyLayer(nn.Module):
    def __init__(self, noise_std, seed=None):
        super(NoisyLayer, self).__init__()
        self.noise_std = noise_std
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(seed)

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        else:
            return x

class net_AE_w_tanh_binaryloss_v2(nn.Module):
    def __init__(self, shape_x=128, M=666, fixed_mask=None):
        super(net_AE_w_tanh_binaryloss_v2, self).__init__()
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
        self.noise_layer = NoisyLayer(noise_std=0.8, seed=2024)
        self.reconstruct = nn.Sequential(
            # conv1
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.4),
            # conv2
            nn.Conv2d(8, 8,3, 2, 1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(0.4),
            # # conv3
            # nn.Conv2d(32, 1, 5, 1, 2),
            # nn.ReLU(),
            # nn.Dropout(0.4),
        )
        self.reconstruct2 = nn.Sequential(
            nn.Linear(2048, shape_x**2),
        )

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.matmul(x, self.tanh(self.mask_mat))
        x = self.noise_layer(x)
        x = self.feature(x)
        x = x.view(-1, 1, self.shape_x, self.shape_x)
        x = self.reconstruct(x)

        x = x.view(x.size(0), -1)
        x = self.reconstruct2(x)
        x = x.view(-1, 1, self.shape_x, self.shape_x)

        return x, self.mask_mat

if __name__ == "__main__":

    shape_x=32
    M=66

    network = net_AE_w_tanh_binaryloss_v2(shape_x=shape_x,M=M)
    input_tensor = torch.randn(10,shape_x,shape_x)
    output_tensor, out_maskmat = network(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
    print("Output mask mat shape:", out_maskmat.shape)

    