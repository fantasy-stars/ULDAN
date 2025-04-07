from torch import nn
import torch

class discriminator_v1(nn.Module):
    def __init__(self,shape_y=32):
        super().__init__()
        self.func = nn.Sequential(
        nn.Linear(shape_y**2, (shape_y**2)//2),
        nn.BatchNorm1d((shape_y**2)//2),
        nn.ReLU(),
        nn.Linear((shape_y**2)//2, shape_y),
        nn.BatchNorm1d(shape_y),
        nn.ReLU(),
        nn.Linear(shape_y, 1)
    )
        
    def forward(self, x):
        
        return self.func(x)


# class discriminator_v1(nn.Module):
#     def __init__(self,shape_y=32):
#         super().__init__()
#         self.func = nn.Sequential(
#         nn.Linear(shape_y**2, 50),
#         nn.ReLU(),
#         nn.Linear(50, 20),
#         nn.ReLU(),
#         nn.Linear(20, 1)
#     )
        
#     def forward(self, x):
        
#         return self.func(x)




