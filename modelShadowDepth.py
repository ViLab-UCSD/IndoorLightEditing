import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def depthToPoint(fov, depth ):
    height, width = depth.size(2), depth.size(3)
    xRange = 1 * np.tan(fov / 2 )
    yRange = float(height) / float(width) * xRange

    x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
            np.linspace(-yRange, yRange, height ) )

    y = np.flip(y, axis=0 )
    z = -np.ones( (height, width), dtype=np.float32 )

    pCoord = np.stack([x, y, z], axis = 0 )[np.newaxis, :]
    pCoord = pCoord.astype(np.float32 )
    pCoord = Variable(torch.from_numpy(pCoord ) ).cuda()
    point = pCoord * depth

    return point

class denoiser(nn.Module ):
    def __init__(self, fov=57.95 ):
        super(denoiser, self).__init__()
        self.fov = fov / 180.0 * np.pi

        # Feature Net
        self.padFeat_1 = nn.ReplicationPad2d(1 )
        self.convFeat_1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1 )
        self.gnFeat_1 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padRaw_1 = nn.ReplicationPad2d(1 )
        self.convRaw_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1 )
        self.gnRaw_1 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padEncode_1 = nn.ReplicationPad2d(1 )
        self.convEncode_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2 )
        self.gnEncode_1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padEncode_2 = nn.ReplicationPad2d(1 )
        self.convEncode_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2 )
        self.gnEncode_2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padEncode_3 = nn.ReplicationPad2d(1 )
        self.convEncode_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2 )
        self.gnEncode_3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padDecode_3 = nn.ReplicationPad2d(1 )
        self.convDecode_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1 )
        self.gnDecode_3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padDecode_2 = nn.ReplicationPad2d(1 )
        self.convDecode_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1 )
        self.gnDecode_2 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padDecode_1 = nn.ReplicationPad2d(1 )
        self.convDecode_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 )
        self.gnDecode_1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padDecode_0 = nn.ReplicationPad2d(1 )
        self.convDecode_0 = nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, stride=1 )
        self.gnDecode_0 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padFinal = nn.ReplicationPad2d(1 )
        self.convFinal = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1 )

    def forward(self, rawInput, normal, depth, confs ):

        point = depthToPoint(self.fov, depth  )

        # Compute the feature from raw input
        xRaw_1 = F.relu(self.gnRaw_1(self.convRaw_1(self.padRaw_1(rawInput ) ) ), True )

        # Compute the feature
        featureInput = torch.cat([normal, point ], dim=1 )
        xFeat_1 = F.relu(self.gnFeat_1(self.convFeat_1(self.padFeat_1(featureInput ) ) ), True )

        # Compute the feature from light source prediction
        encoderInput = torch.cat([xRaw_1, xFeat_1], dim=1 )
        x1 = F.relu(self.gnEncode_1(self.convEncode_1(self.padEncode_1(encoderInput ) ) ), True )
        x2 = F.relu(self.gnEncode_2(self.convEncode_2(self.padEncode_2(x1 ) ) ), True )
        x3 = F.relu(self.gnEncode_3(self.convEncode_3(self.padEncode_3(x2 ) ) ), True )

        dx3 = F.relu(self.gnDecode_3(self.convDecode_3(self.padDecode_3(x3 ) ) ), True )

        dx3 = F.interpolate(dx3, [x2.size(2), x2.size(3) ], mode='bilinear')
        dx3 = torch.cat([dx3, x2], dim=1 )
        dx2 = F.relu(self.gnDecode_2(self.convDecode_2(self.padDecode_2(dx3 ) ) ), True )

        dx2 = F.interpolate(dx2, [x1.size(2), x1.size(3) ], mode='bilinear')
        dx2 = torch.cat([dx2, x1], dim=1 )
        dx1 = F.relu(self.gnDecode_1(self.convDecode_1(self.padDecode_1(dx2 ) ) ) )

        dx1 = F.interpolate(dx1, [xRaw_1.size(2), xRaw_1.size(3)], mode='bilinear')
        dx1 = torch.cat([dx1, xRaw_1], dim=1 )
        dx0 = F.relu(self.gnDecode_0(self.convDecode_0(self.padDecode_0(dx1 ) ) ) )

        xFinal = torch.sigmoid(self.convFinal(self.padFinal(dx0 ) ) )
        xFinal = torch.clamp(xFinal * confs + (1 - confs) * torch.clamp(rawInput, 0, 1), 0, 1 )

        return xFinal
