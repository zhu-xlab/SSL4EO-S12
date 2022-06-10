import torch
import torch.nn as nn
import torch.nn.functional as F


def get_segmentation_model(backbone, feature_indices, feature_channels):
    """Creates a UNet from a pretrained backbone

    Args:
        backbone (torch.nn.Module): Pre-trained backbone in the form of "Sequential"
        feature_indices (list(int)): Indices in the Sequential backbone from which to extract intermediate features
        feature_channels ([type]): Number of channels per feature extracted

    Returns:
        [type]: [description]
    """
    model = SegmentationEncoder(backbone, feature_indices, diff=True)
    unet = UNet(model, feature_channels, 1, bilinear=True, concat_mult=1, dropout_rate=0.3)
    # unet = UNetSmall(model, feature_channels, 1, bilinear=True, concat_mult=1)
    return unet


class SegmentationEncoder(torch.nn.Module):
    def __init__(self, backbone, feature_indices, diff=False):
        super().__init__()
        self.feature_indices = list(sorted(feature_indices))

        # # A number of channels for each encoder feature tensor, list of integers
        # self._out_channels = feature_channels  # [3, 16, 64, 128, 256, 512]

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels = 13

        # Define encoder modules below
        self.encoder = backbone

        self.diff = diff

    def forward(self, x1, x2):
        """Produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
        """
        feats = [self.concatenate(x1, x2)]
        for i, module in enumerate(self.encoder.children()):
            x1 = module(x1)
            x2 = module(x2)
            if i in self.feature_indices:
                feats.append(self.concatenate(x1, x2))
            if i == self.feature_indices[-1]:
                break

        return feats

    def concatenate(self, x1, x2):
        if self.diff:
            return torch.abs(x1 - x2)
        else:
            torch.cat([x1, x2], 1)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, encoder, feature_channels, n_classes, concat_mult=2, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         factor = 2 if bilinear else 1
#         self.feature_channels = feature_channels
#         for i in range(0, len(feature_channels)-1):
#             in_ch = feature_channels[i + 1] * concat_mult + feature_channels[i] * concat_mult
#             setattr(self, "up_%d" %i, Up(in_ch, feature_channels[i] * concat_mult, bilinear))
#         self.outc = OutConv(feature_channels[0] * concat_mult, n_classes)
#         self.encoder = encoder
#
#     def forward(self, *in_x):
#         features = self.encoder(*in_x)
#         features = features[1:]
#         x = features[-1]
#         for i in range(len(features) - 2, -1, -1):
#             up = getattr(self, 'up_%d' %i)
#             x = up(x, features[i])
#         logits = self.outc(x)
#         return logits


class UNet(nn.Module):
    def __init__(self, encoder, feature_channels, n_classes, concat_mult=2, bilinear=True, dropout_rate=0.5):
        """Simple segmentation network

        Args:
            encoder (torch Sequential): The pre-trained encoder
            feature_channels (list(int)): Number of channels per input feature
            n_classes (int): output number of classes
            concat_mult (int, optional): The amount of features being fused. Defaults to 2.
            bilinear (bool, optional): If use bilinear interpolation (I have defaulted to nearest since it has been shown to be better sometimes). Defaults to True.
        """
        super(UNet, self).__init__()
        self.n_classes = n_classes  # 1
        self.bilinear = bilinear
        # factor = 2 if bilinear else 1
        self.feature_channels = feature_channels 
        self.dropout = torch.nn.Dropout2d(dropout_rate)
        for i in range(0, len(feature_channels) - 1):
            if i == len(feature_channels) - 2:
                in_ch = feature_channels[i + 1] * concat_mult
            else:
                in_ch = feature_channels[i + 1] * concat_mult
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(in_ch, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1))
            setattr(self, 'shrink2%d' % i,
                    nn.Conv2d(feature_channels[i] * concat_mult * 2, feature_channels[i] * concat_mult, kernel_size=3, stride=1, padding=1, bias=False))
            setattr(self, 'batchnorm%d' % i,
                    nn.BatchNorm2d(feature_channels[i] * concat_mult))
        self.outc = OutConv(feature_channels[0] * concat_mult, n_classes)
        self.encoder = encoder

    def forward(self, *in_x):
        features = self.encoder(*in_x)
        features = features[1:]
        x = features[-1]
        for i in range(len(features) - 2, -1, -1):
            conv = getattr(self, 'shrink%d' % i)
            x = F.upsample_nearest(x, scale_factor=2)
            x = conv(x)
            if features[i].shape[-1] != x.shape[-1]:
                x2 = F.upsample_nearest(features[i], scale_factor=2)
            else:
                x2 = features[i]
            x = torch.cat([x, x2], 1)
            conv2 = getattr(self, 'shrink2%d' % i)
            x = conv2(x)
            bn = getattr(self, 'batchnorm%d' % i)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        x = F.upsample_nearest(x, scale_factor=2)
        logits = self.outc(x)
        return logits


class UNetSmall(nn.Module):
    def __init__(self, encoder, feature_channels, n_classes, concat_mult=2, bilinear=True):
        """Simple segmentation network

        Args:
            encoder (torch Sequential): The pre-trained encoder
            feature_channels (list(int)): Number of channels per input feature
            n_classes (int): output number of classes
            concat_mult (int, optional): The amount of features being fused. Defaults to 2.
            bilinear (bool, optional): If use bilinear interpolation (I have defaulted to nearest since it has been shown to be better sometimes). Defaults to True.
        """
        super(UNetSmall, self).__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.feature_channels = feature_channels
        for i in range(0, len(feature_channels)):
            setattr(self, 'shrink%d' % i,
                    nn.Conv2d(feature_channels[i], feature_channels[0], kernel_size=1, stride=1, padding=0))
        
        self.aggregate = nn.Conv2d(len(feature_channels) * feature_channels[0], feature_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(feature_channels[0])
        self.outc = OutConv(feature_channels[0], n_classes)
        self.encoder = encoder

    def forward(self, *in_x):
        features = self.encoder(*in_x)
        b, c, h, w = in_x[0].shape
        features = features[1:]
        ret = []
        for i in range(len(features)):
            conv = getattr(self, 'shrink%d' % i)
            x = conv(features[i])
            ratio = h // features[i].shape[-2]
            ret.append(F.upsample_bilinear(x, scale_factor=ratio))
        ret = torch.cat(ret, 1)
        ret = self.aggregate(ret)
        ret = self.bn(ret)
        ret = F.relu(ret, True)
        logits = self.outc(ret)
        return logits
