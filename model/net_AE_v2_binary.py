import torch.nn as nn
import torch
import torch
import torch.nn as nn

class NoisyLayer(nn.Module):
    def __init__(self, noise_std, seed=None):
        super(NoisyLayer, self).__init__()
        self.noise_std = noise_std
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(seed)

    def forward(self, x):
        if self.training:  # 只在训练时添加噪声
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        else:
            return x

# 之前net_AE_w_tanh_binaryloss_v1对于随机图案性能不好
# 这里v2来调整网络结构
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
        self.noise_layer = NoisyLayer(noise_std=0.8, seed=2024)  # 添加噪声层，并设置种子
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
        '''
            model_input = Input(shape=input_shape)

    x = Conv2D(8, kernel_size=(3, 3), activation='relu')(model_input)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x) #

    x = Conv2D(8, kernel_size=(3, 3), strides=2, activation='relu')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(outsize, activation='sigmoid')(x)

    cnn = Model(inputs=model_input, outputs=x)
        '''
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
    # 测试网络结构
    shape_x=32
    M=66

    network = net_AE_w_tanh_binaryloss_v2(shape_x=shape_x,M=M)
    input_tensor = torch.randn(10,shape_x,shape_x)  # 输入张量的示例，大小为[batch_size, channels, height, width]
    output_tensor, out_maskmat = network(input_tensor)
    print("Output tensor shape:", output_tensor.shape)  # 输出张量的形状
    print("Output mask mat shape:", out_maskmat.shape)