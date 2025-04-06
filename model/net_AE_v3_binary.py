import torch.nn as nn
import torch
import torch.nn.init as init

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d) or isinstance(
                    m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(
                    m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class UpsampleCat(nn.Module):
    def __init__(self, in_nc, out_nc):
        super(UpsampleCat, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc

        self.deconv = nn.ConvTranspose2d(in_nc, out_nc, 2, 2, 0, bias=False)
        initialize_weights(self.deconv, 0.1)

    def forward(self, x1, x2):
        x1 = self.deconv(x1)
        return torch.cat([x1, x2], dim=1)


def conv_func(x, conv, blindspot):
    size = conv.kernel_size[0]
    if blindspot:
        assert (size % 2) == 1
    ofs = 0 if (not blindspot) else size // 2

    if ofs > 0:
        # (padding_left, padding_right, padding_top, padding_bottom)
        pad = nn.ConstantPad2d(padding=(0, 0, ofs, 0), value=0)
        x = pad(x)
    x = conv(x)
    if ofs > 0:
        x = x[:, :, :-ofs, :]
    return x


def pool_func(x, pool, blindspot):
    if blindspot:
        pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
        x = pad(x[:, :, :-1, :])
    x = pool(x)
    return x


def rotate(x, angle):
    if angle == 0:
        return x
    elif angle == 90:
        return torch.rot90(x, k=1, dims=(3, 2))
    elif angle == 180:
        return torch.rot90(x, k=2, dims=(3, 2))
    elif angle == 270:
        return torch.rot90(x, k=3, dims=(3, 2))


class UNet(nn.Module):
    def __init__(self,
                 in_nc=3,
                 out_nc=3,
                 n_feature=48,
                 blindspot=False,
                 zero_last=False):
        super(UNet, self).__init__()
        self.in_nc = in_nc
        self.out_nc = out_nc
        self.n_feature = n_feature
        self.blindspot = blindspot
        self.zero_last = zero_last
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Encoder part
        self.enc_conv0 = nn.Conv2d(self.in_nc, self.n_feature, 3, 1, 1)
        self.enc_conv1 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv0, 0.1)
        initialize_weights(self.enc_conv1, 0.1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc_conv2 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv2, 0.1)
        self.pool2 = nn.MaxPool2d(2)

        self.enc_conv3 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv3, 0.1)
        self.pool3 = nn.MaxPool2d(2)

        self.enc_conv4 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv4, 0.1)
        self.pool4 = nn.MaxPool2d(2)

        self.enc_conv5 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv5, 0.1)
        self.pool5 = nn.MaxPool2d(2)

        self.enc_conv6 = nn.Conv2d(self.n_feature, self.n_feature, 3, 1, 1)
        initialize_weights(self.enc_conv6, 0.1)

        # Decoder part
        self.up5 = UpsampleCat(self.n_feature, self.n_feature)
        self.dec_conv5a = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv5b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv5a, 0.1)
        initialize_weights(self.dec_conv5b, 0.1)

        self.up4 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv4a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv4b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv4a, 0.1)
        initialize_weights(self.dec_conv4b, 0.1)

        self.up3 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv3a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv3b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv3a, 0.1)
        initialize_weights(self.dec_conv3b, 0.1)

        self.up2 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)
        self.dec_conv2a = nn.Conv2d(self.n_feature * 3, self.n_feature * 2, 3,
                                    1, 1)
        self.dec_conv2b = nn.Conv2d(self.n_feature * 2, self.n_feature * 2, 3,
                                    1, 1)
        initialize_weights(self.dec_conv2a, 0.1)
        initialize_weights(self.dec_conv2b, 0.1)

        self.up1 = UpsampleCat(self.n_feature * 2, self.n_feature * 2)

        # Output stages
        self.dec_conv1a = nn.Conv2d(self.n_feature * 2 + self.in_nc, 96, 3, 1,
                                    1)
        initialize_weights(self.dec_conv1a, 0.1)
        self.dec_conv1b = nn.Conv2d(96, 96, 3, 1, 1)
        initialize_weights(self.dec_conv1b, 0.1)
        if blindspot:
            self.nin_a = nn.Conv2d(96 * 4, 96 * 4, 1, 1, 0)
            self.nin_b = nn.Conv2d(96 * 4, 96, 1, 1, 0)
        else:
            self.nin_a = nn.Conv2d(96, 96, 1, 1, 0)
            self.nin_b = nn.Conv2d(96, 96, 1, 1, 0)
        initialize_weights(self.nin_a, 0.1)
        initialize_weights(self.nin_b, 0.1)
        self.nin_c = nn.Conv2d(96, self.out_nc, 1, 1, 0)
        if not self.zero_last:
            initialize_weights(self.nin_c, 0.1)

    def forward(self, x):
        # Input stage
        blindspot = self.blindspot
        if blindspot:
            x = torch.cat([rotate(x, a) for a in [0, 90, 180, 270]], dim=0)
        # Encoder part
        pool0 = x
        x = self.act(conv_func(x, self.enc_conv0, blindspot))
        x = self.act(conv_func(x, self.enc_conv1, blindspot))
        x = pool_func(x, self.pool1, blindspot)
        pool1 = x

        x = self.act(conv_func(x, self.enc_conv2, blindspot))
        x = pool_func(x, self.pool2, blindspot)
        pool2 = x

        x = self.act(conv_func(x, self.enc_conv3, blindspot))
        x = pool_func(x, self.pool3, blindspot)
        pool3 = x

        x = self.act(conv_func(x, self.enc_conv4, blindspot))
        x = pool_func(x, self.pool4, blindspot)
        pool4 = x

        x = self.act(conv_func(x, self.enc_conv5, blindspot))
        x = pool_func(x, self.pool5, blindspot)

        x = self.act(conv_func(x, self.enc_conv6, blindspot))

        # Decoder part
        x = self.up5(x, pool4)
        x = self.act(conv_func(x, self.dec_conv5a, blindspot))
        x = self.act(conv_func(x, self.dec_conv5b, blindspot))

        x = self.up4(x, pool3)
        x = self.act(conv_func(x, self.dec_conv4a, blindspot))
        x = self.act(conv_func(x, self.dec_conv4b, blindspot))

        x = self.up3(x, pool2)
        x = self.act(conv_func(x, self.dec_conv3a, blindspot))
        x = self.act(conv_func(x, self.dec_conv3b, blindspot))

        x = self.up2(x, pool1)
        x = self.act(conv_func(x, self.dec_conv2a, blindspot))
        x = self.act(conv_func(x, self.dec_conv2b, blindspot))

        x = self.up1(x, pool0)

        # Output stage
        if blindspot:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            pad = nn.ConstantPad2d(padding=(0, 0, 1, 0), value=0)
            x = pad(x[:, :, :-1, :])
            x = torch.split(x, split_size_or_sections=x.shape[0] // 4, dim=0)
            x = [rotate(y, a) for y, a in zip(x, [0, 270, 180, 90])]
            x = torch.cat(x, dim=1)
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        else:
            x = self.act(conv_func(x, self.dec_conv1a, blindspot))
            x = self.act(conv_func(x, self.dec_conv1b, blindspot))
            x = self.act(conv_func(x, self.nin_a, blindspot))
            x = self.act(conv_func(x, self.nin_b, blindspot))
            x = conv_func(x, self.nin_c, blindspot)
        return x


# if __name__ == "__main__":
#     import numpy as np
#     x = torch.from_numpy(np.zeros((10, 1, 32, 32), dtype=np.float32))
#     print(x.shape)
#     net = UNet(in_nc=1, out_nc=1, blindspot=False)
#     y = net(x)
#     print(y.shape)
def awgn_noise(signal, snr_db):
    # 计算信号的功率
    signal_power = torch.mean(torch.abs(signal) ** 2)    
    # 计算噪声的功率
    noise_power = signal_power / (10 ** (snr_db / 10))    
    # 生成符合高斯分布的随机噪声
    noise = torch.sqrt(noise_power)*torch.randn(signal.shape).to(device)#torch.normal(0, torch.sqrt(noise_power), signal.shape)    

    return noise

class NoisyLayer(nn.Module):
    def __init__(self, noise_std, snr_db=None, seed=None):
        super(NoisyLayer, self).__init__()
        self.noise_std = noise_std
        self.seed = seed
        self.snr_db=snr_db
        if self.seed is not None:
            torch.manual_seed(seed)

    def forward(self, x):
        if self.training:  # 只在训练时添加噪声
            # print('self.training')
            # noise = torch.randn_like(x) * self.noise_std
            noise=torch.zeros_like(x)
            if self.snr_db is not None:
                noise+=awgn_noise(x,self.snr_db)
            return x + noise
        else:
            return x

# 之前net_AE_w_tanh_binaryloss_v1对于随机图案性能不好
# 这里v2来调整网络结构
class net_AE_w_tanh_binaryloss_v3(nn.Module):
    def __init__(self, shape_x=128,shape_x2=None, M=666, fixed_mask=None, mask_trainable=False):
        super(net_AE_w_tanh_binaryloss_v3, self).__init__()
        self.shape_x = shape_x
        if shape_x2==None:
            self.shape_x2=shape_x
        else:
            self.shape_x2=shape_x2        
        
        if fixed_mask is None:
            self.mask_mat = nn.Parameter(torch.randn(shape_x*shape_x2, M))
            # self.mask_mat = nn.Parameter(torch.zeros(shape_x**2, M))
        elif (fixed_mask is not None) and mask_trainable:
            # self.mask_mat = torch.tensor(fixed_mask, requires_grad=True)
            self.mask_mat=fixed_mask.clone().detach().requires_grad_(True)
        elif (fixed_mask is not None) and  (not mask_trainable):
            self.mask_mat = torch.tensor(fixed_mask, requires_grad=False)
        else:
            raise ValueError('fixed_mask和mask_trainable参数设置错误！')

        self.feature = nn.Sequential(
            nn.Linear(M, shape_x*shape_x2),
            nn.ReLU(),
            nn.BatchNorm1d(shape_x*shape_x2),  
        )
        # 备忘： (noise_std=0.8, snr_db=None, seed=2024) 
        self.noise_layer = NoisyLayer(noise_std=0.8, snr_db=50, seed=2024)  # 添加噪声层，并设置种子

        self.reconstruct = UNet(in_nc=1, out_nc=1, blindspot=False)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.matmul(x, self.tanh(self.mask_mat))
        x = self.noise_layer(x)
        x = self.feature(x)
        x = x.view(-1, 1, self.shape_x, self.shape_x2)
        x = self.reconstruct(x)

        return x, self.mask_mat



if __name__ == "__main__":
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 测试网络结构
    shape_x=32
    shape_x2=64
    M=66

    network = net_AE_w_tanh_binaryloss_v3(shape_x=shape_x,shape_x2=shape_x2,M=M).to(device)
    input_tensor = torch.randn(10,shape_x,shape_x2).to(device)  # 输入张量的示例，大小为[batch_size, channels, height, width]
    output_tensor, out_maskmat = network(input_tensor)
    print("Output tensor shape:", output_tensor.shape)  # 输出张量的形状
    print("Output mask mat shape:", out_maskmat.shape)