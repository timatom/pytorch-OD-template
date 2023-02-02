import torch.nn as nn

class FPNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, groups=1, norm_layer=None):
        super(FPNBlock, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, targets):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        if targets is not None:
            return x, targets
        
        return x

class FPNBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FPNBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, targets=None):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)
        
        if targets is not None:
            return out, targets

        return out
