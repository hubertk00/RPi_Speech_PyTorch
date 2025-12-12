import torch
import torch.nn as nn
import torch.nn.functional as F

class SubSpectralNorm(nn.Module):
    def __init__(self, channels, sub_bands, eps=1e-5):
        super().__init__()
        self.sub_bands = sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands, eps=eps)

    def forward(self, x):
        N, C, F, T = x.size()
        if F % self.sub_bands != 0:
            return F.batch_norm(x, self.bn.running_mean[:C], self.bn.running_var[:C], 
                                self.bn.weight[:C], self.bn.bias[:C], 
                                self.training, self.bn.momentum, self.bn.eps)
        x = x.view(N, C * self.sub_bands, F // self.sub_bands, T)
        x = self.bn(x)
        x = x.view(N, C, F, T)
        return x

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane,
        out_plane,
        idx,
        kernel_size=3,
        stride=1,
        groups=1,
        use_dilation=False,
        activation=True,
        swish=False,
        BN=True,
        ssn=False,
    ):
        super().__init__()

        def get_padding(kernel_size, use_dilation):
            rate = 1
            padding_len = (kernel_size - 1) // 2
            if use_dilation and kernel_size > 1:
                rate = int(2**self.idx)
                padding_len = rate * padding_len
            return padding_len, rate

        self.idx = idx

        if isinstance(kernel_size, (list, tuple)):
            padding = []
            rate = []
            for k_size in kernel_size:
                temp_padding, temp_rate = get_padding(k_size, use_dilation)
                rate.append(temp_rate)
                padding.append(temp_padding)
        else:
            padding, rate = get_padding(kernel_size, use_dilation)

        layers = []
        layers.append(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding, dilation=rate, groups=groups, bias=False)
        )
        
        if ssn:
            layers.append(SubSpectralNorm(out_plane, sub_bands=1)) 
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
            
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class BCResBlock(nn.Module):
    def __init__(self, in_plane, out_plane, idx, stride):
        super().__init__()
        self.transition_block = in_plane != out_plane
        kernel_size = (3, 3)

        layers = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (kernel_size[0], 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, 1))

        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, kernel_size[1]),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.avg_gpool(x)

        x = self.f1(x)
        x = x + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        x = F.relu(x, True)
        return x

def BCBlockStage(num_layers, last_channel, cur_channel, idx, use_stride):
    stage = nn.ModuleList()
    channels = [last_channel] + [cur_channel] * num_layers
    for i in range(num_layers):
        stride = (2, 1) if use_stride and i == 0 else (1, 1)
        stage.append(BCResBlock(channels[i], channels[i + 1], idx, stride))
    return stage

class BCResNets(nn.Module):
    def __init__(self, base_c, layers, num_classes=12):
        super().__init__()
        self.num_classes = num_classes
        self.n = layers
        
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            base_c * 4,
            base_c * 4
        ]
        
        self.s = [1, 2] 
        self._build_network()

    def _build_network(self):
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (1, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        
        self.BCBlocks = nn.ModuleList([])
        for idx, n in enumerate(self.n):
            use_stride = idx in self.s
            self.BCBlocks.append(BCBlockStage(n, self.c[idx], self.c[idx + 1], idx, use_stride))

        last_stage_out = self.c[len(self.n)] 
        hidden_dim = self.c[len(self.n) + 1]

        self.classifier = nn.Sequential(
            nn.Conv2d(
                last_stage_out, last_stage_out, (5, 5), bias=False, groups=last_stage_out, padding=(0, 2)
            ),
            nn.Conv2d(last_stage_out, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(hidden_dim, self.num_classes, 1),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.cnn_head(x)
        for i, num_modules in enumerate(self.n):
            for j in range(num_modules):
                x = self.BCBlocks[i][j](x)
        x = self.classifier(x)
        x = x.view(-1, x.shape[1])
        return x

def ResNet8(input_channels=20, num_classes=9, k=1.5):
    base_c = int(8 * k)
    return BCResNets(base_c, layers=[1, 1, 1], num_classes=num_classes)

def ResNet14(input_channels=20, num_classes=9, k=1.5):
    base_c = int(8 * k)
    return BCResNets(base_c, layers=[2, 2, 2], num_classes=num_classes)