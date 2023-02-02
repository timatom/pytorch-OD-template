import torch.nn as nn

class ResNetBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        '''
        ResNet Block.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
            downsample (nn.Module): Downsample module.
            groups (int): Number of groups.
            base_width (int): Base width.
            dilation (int): Dilation.
            norm_layer (nn.Module): Normalization layer.
        '''
        super(ResNetBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        width = int(out_channels * (base_width / 64.)) * groups
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn2 = norm_layer(out_channels * self.expansion)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, targets=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        if targets is not None:
            return out, targets
        
        return out


class ResNetBottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm_layer=None):
        '''
        ResNet Bottleneck.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride of the convolution.
            downsample (nn.Module): Downsample module.
            norm_layer (nn.Module): Normalization layer.
        '''
        super(ResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = norm_layer(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, targets=None):
        '''
        Forward pass.
        Args:
            x (torch.Tensor): Input tensor.
        '''
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        if targets is not None:
            return out, targets

        return out
