import torch
import torch.nn as nn
import torch.nn.functional as F

class TCSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(TCSConv, self).__init__()
        self.depthwise_conv = nn.Conv1d(in_channels, in_channels, kernel_size, groups=in_channels, padding='same')
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

class SubBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SubBlock, self).__init__()
        self.tcs_conv = TCSConv(in_channels, out_channels, kernel_size)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x, residual=None):
        x = self.tcs_conv(x)
        x = self.bnorm(x)
        if residual is not None:
            x = x + residual
        x = F.relu(x)
        x = self.dropout(x)
        return x

class MainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, R=1):
        super(MainBlock, self).__init__()
        self.residual_pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.residual_batchnorm = nn.BatchNorm1d(out_channels)
        self.sub_blocks = nn.ModuleList()
        self.sub_blocks.append(SubBlock(in_channels, out_channels, kernel_size))
        for i in range(R-1):
            self.sub_blocks.append(SubBlock(out_channels, out_channels, kernel_size))
            
    def forward(self, x):
        residual = self.residual_pointwise(x)
        residual = self.residual_batchnorm(residual)
        for i, layer in enumerate(self.sub_blocks):
            if (i+1) == len(self.sub_blocks):
                x = layer(x, residual)
            else:
                x = layer(x)
        return x

class MatchboxNet(nn.Module):
    def __init__(self, input_channels=20, num_classes=1, B=3, R=1, C=64):
        super(MatchboxNet, self).__init__()
        self.input_norm = nn.InstanceNorm1d(input_channels, affine=True)

        kernel_sizes = [k*2+11 for k in range(1, B+1)] 

        self.prologue_conv1 = nn.Conv1d(input_channels, 128, kernel_size=11, stride=2, padding=5) 
        self.prologue_bnorm1 = nn.BatchNorm1d(128)

        self.blocks = nn.ModuleList()
        self.blocks.append(MainBlock(128, C, kernel_sizes[0], R=R))
        
        for i in range(1, B):
            self.blocks.append(MainBlock(C, C, kernel_size=kernel_sizes[i], R=R))

        self.epilogue_conv1 = nn.Conv1d(C, 128, kernel_size=29, dilation=2, padding=28)
        self.epilogue_bnorm1 = nn.BatchNorm1d(128)

        self.epilogue_conv2 = nn.Conv1d(128, 128, kernel_size=1)
        self.epilogue_bnorm2 = nn.BatchNorm1d(128)

        self.epilogue_conv3 = nn.Conv1d(128, num_classes, kernel_size=1)
        
        self.epilogue_adaptivepool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: [Batch, n_mfcc, Time]
        x = self.input_norm(x)
        x = self.prologue_conv1(x)
        x = self.prologue_bnorm1(x)
        x = F.relu(x)

        for layer in self.blocks:
            x = layer(x)

        x = self.epilogue_conv1(x)
        x = self.epilogue_bnorm1(x)
        x = F.relu(x) 

        x = self.epilogue_conv2(x)
        x = self.epilogue_bnorm2(x)
        x = F.relu(x)

        x = self.epilogue_conv3(x)
        x = self.epilogue_adaptivepool(x)
        
        x = x.view(x.size(0), -1) 
        
        return x