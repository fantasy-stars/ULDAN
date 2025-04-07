import torch.nn as nn
import torch
import torch
import torch.nn as nn

class target_net_decoder_v1(nn.Module):
    def __init__(self,M=666, shape_y=32):
        super(target_net_decoder_v1, self).__init__()
        self.shape_y = shape_y
        
        self.feature = nn.Sequential(
            nn.Linear(M, shape_y**2),
            nn.ReLU(),
            nn.BatchNorm1d(shape_y**2),  
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

    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1, 1, self.shape_y, self.shape_y)
        x = self.reconstruct(x)

        return x

if __name__ == "__main__":

    shape_y=32
    M=66

    network = target_net_decoder_v1(M=M,shape_y=shape_y)
    input_tensor = torch.randn(10,M)
    output_tensor = network(input_tensor)
    print("Output tensor shape:", output_tensor.shape)
