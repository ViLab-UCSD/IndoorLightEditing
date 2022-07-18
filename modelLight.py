import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


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
    pCoord = torch.from_numpy(pCoord ).cuda()
    point = pCoord * depth

    return point


def mask2center(mask, pCoord, depth, isWin ):
    pCoord = pCoord / np.sqrt(
        np.maximum(np.sum(pCoord  * pCoord, axis=0, keepdims=True), 1e-6) )
    direc = np.sum(np.sum(mask * pCoord, axis=1), axis=1)  / np.sum(mask )
    direc = direc / np.sqrt(np.maximum(np.sum(direc * direc ), 1e-6 ) )

    depth = depth.reshape(-1 )
    mask  = mask.reshape(-1 )
    if isWin:
        depthCandi = depth[mask == 0 ]
    else:
        depthCandi = depth[mask > 0.9 ]
    depth = np.median(depthCandi )

    center = direc / -direc[2] * depth
    return  center, direc


def crop2Dregion(masks, depths, isWin, fov, newWidth, newHeight):
    masks = masks.cpu().numpy()
    depths = depths.cpu().numpy()

    batchSize = masks.shape[0]
    height, width = masks.shape[2:]
    xRange = 1 * np.tan(fov / 2 )
    yRange = float(height) / float(width) * xRange

    x, y = np.meshgrid(np.linspace(-xRange, xRange, width ),
            np.linspace(-yRange, yRange, height ) )

    y = np.flip(y, axis=0 )
    z = -np.ones( (height, width), dtype=np.float32 )

    pCoord = np.stack([x, y, z], axis = 0 ).astype(np.float32 )

    featureArr, centerArr, direcArr = [], [], []
    for n in range(0, batchSize ):
        mask = masks[n:n+1, :].squeeze()
        depth = depths[n:n+1, :].squeeze()

        # Crop according to the mask
        maskRow = np.sum(mask, axis=1 )
        maskCol = np.sum(mask, axis=0 )

        rowId = np.nonzero(maskRow )[0]
        colId = np.nonzero(maskCol )[0]

        rs, re = rowId.min(), rowId.max()
        cs, ce = colId.min(), colId.max()

        rs = max(rs - 3, 0 )
        re = min(re + 3, height )
        cs = max(cs - 3, 0 )
        ce = min(ce + 3, width )

        cropHeight = re - rs
        cropWidth = ce  - cs

        mask = mask[rs:re, cs:ce]
        depth = depth[rs:re, cs:ce]

        pCoord_cropped = pCoord[:, rs:re, cs:ce ]

        center, direc = mask2center(mask, pCoord_cropped, depth, isWin )
        center = center.reshape(1, 3, 1, 1 )
        direc = direc.reshape(1, 3, 1, 1 )

        centerMap = np.zeros((1, 3, newHeight, newWidth ), dtype = np.float32 ) + center
        direcMap = np.zeros((1, 3, newHeight, newWidth ), dtype = np.float32 ) + direc

        center = torch.from_numpy(center ).cuda()
        direc = torch.from_numpy(direc ).cuda()
        centerMap = torch.from_numpy(centerMap ).cuda()
        direcMap = torch.from_numpy(direcMap ).cuda()

        centerArr.append(center )
        direcArr.append(direc )

        feature = torch.cat([centerMap, direcMap ], dim=1 )

        featureArr.append(feature )

    featureArr = torch.cat(featureArr, dim=0 )
    centerArr = torch.cat(centerArr, dim=0 )
    direcArr = torch.cat(direcArr, dim=0 )

    return featureArr, centerArr, direcArr


class lightSrcEncoder(nn.Module ):
    def __init__(self, isInv, isFg ):
        super(lightSrcEncoder, self).__init__()
        self.pad1 = nn.ReplicationPad2d(1 )
        self.isInv = isInv
        if isInv:
            self.conv1 = nn.Conv2d(in_channels=10, out_channels=128, kernel_size=4, stride=2, bias =True )
        else:
            self.conv1 = nn.Conv2d(in_channels=17, out_channels=128, kernel_size=4, stride=2, bias =True )
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.pad2 = nn.ReplicationPad2d(1 )
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2, bias=True )
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.pad3 = nn.ReplicationPad2d(1 )
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels=256, kernel_size=4, stride=2, bias=True )
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.pad4 = nn.ReplicationPad2d(1 )
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, bias=True )
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.pad5 = nn.ReplicationPad2d(1 )
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True )
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True )
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024 )

    def forward(self, inputs ) :
        x1 = F.relu(self.gn1(self.conv1(self.pad1(inputs ) ) ), True )
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True )
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True )
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True )
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4 ) ) ), True )
        x6 = F.relu(self.gn6(self.conv6(x5 ) ), True )
        x6 = torch.mean(torch.mean(x6, dim=2, keepdim=True ), dim=3, keepdim=True )
        return x6


class lampNet(nn.Module ):
    def __init__(self, isInv, fov=57.95 ):
        super(lampNet, self).__init__()
        # Predict the light source parameters of the visible light sources
        self.isInv = isInv
        self.fov = fov / 180.0 * np.pi

        self.encoder = lightSrcEncoder(isInv, isFg = False)
        if  self.isInv:
            self.convBox_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
            self.gnBox = nn.GroupNorm(num_groups=16, num_channels=256 )
            self.convBox_2 = nn.Conv2d(in_channels=256, out_channels=9, kernel_size=1, stride=1, bias=True )
        else:
            self.convCenter_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
            self.gnCenter = nn.GroupNorm(num_groups=16, num_channels=256 )
            self.convCenter_2 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, bias=True )

        self.convInt_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
        self.gnInt = nn.GroupNorm(num_groups=16, num_channels=256 )
        self.convInt_2 = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1, stride=1, bias=True )

    def fromEulerAngleToMatrix(self, alpha, beta, gamma ):
        batchSize = alpha.size(0 )
        zeros = torch.zeros(alpha.size(), dtype=torch.float32 ).cuda()
        ones = torch.ones(alpha.size(), dtype=torch.float32 ).cuda()
        cosAlpha = torch.cos(alpha )
        sinAlpha = torch.sin(alpha )
        cosBeta = torch.cos(beta )
        sinBeta = torch.sin(beta )
        cosGamma = torch.cos(gamma )
        sinGamma = torch.sin(gamma )

        zMat = torch.cat([cosGamma, sinGamma, zeros,
            -sinGamma, cosGamma, zeros,
            zeros, zeros, ones ], dim=1).view(batchSize, 3, 3)
        yMat = torch.cat([ones, zeros, zeros,
            zeros, cosBeta, sinBeta,
            zeros, -sinBeta, cosBeta ], dim=1).view(batchSize, 3, 3)
        xMat = torch.cat([cosAlpha, sinAlpha, zeros,
            -sinAlpha, cosAlpha, zeros,
            zeros, zeros, ones ], dim=1).view(batchSize, 3, 3)
        rotMatrix = torch.matmul(torch.matmul(zMat, yMat ), xMat )
        return rotMatrix

    def computeInvCenter(self, centerRaw, height, width ):
        theta, phi, dist = torch.split(centerRaw, [1, 1, 1], dim=1 )

        zAxis = torch.FloatTensor([0, 0, 1] ).cuda().reshape(1, 3)
        yAxis = torch.FloatTensor([0, 1, 0] ).cuda().reshape(1, 3)
        xAxis = torch.FloatTensor([1, 0, 0] ).cuda().reshape(1, 3)

        tanx = np.tan(self.fov / 2.0 )
        tany = tanx / float(width ) * float(height )
        fovy = np.arctan(tany ) * 2.0

        theta = torch.sigmoid(theta ) * (np.pi - fovy )
        phi = torch.tanh(phi ) * np.pi
        dist = torch.exp(dist )

        centerDirec = torch.sin(theta ) * torch.cos(phi ) * xAxis\
            + torch.sin(theta ) * torch.sin(phi ) * yAxis \
            + torch.cos(theta ) * zAxis
        center = centerDirec * dist

        return center


    def forward(self, im, depth, albedo,
                lightOnMasks, lightOnMask=None ):

        batchSize = im.size(0 )
        point = depthToPoint(self.fov, depth )

        if self.isInv:
            inputs = torch.cat([im, point, albedo,
                                lightOnMasks], dim=1 )
        else:
            cropInputs, center, centerDirec = crop2Dregion(
                lightOnMask, depth, isWin=False, fov = self.fov,
                newWidth = depth.size(3), newHeight = depth.size(2) )

            inputs = torch.cat([im, point, albedo,
                                lightOnMask, lightOnMasks, cropInputs ], dim=1 )

        x = self.encoder(inputs )
        if  self.isInv:
            x_box, x_src = torch.split(x, [512, 512], dim=1 )

            x_box = F.relu(self.gnBox(self.convBox_1(x_box ) ), True )
            x_box = self.convBox_2(x_box )
            x_box = x_box.view(batchSize, 9)
            alpha, beta, gamma, boxLen, center \
                    = torch.split(x_box, [1, 1, 1, 3, 3], dim=1 )

            center = self.computeInvCenter(center, height = depth.size(2), width = depth.size(3 ) )

            alpha = torch.sigmoid(alpha ) * np.pi * 2
            beta = torch.sigmoid(beta ) * np.pi
            gamma = torch.sigmoid(gamma ) * np.pi * 2
            axes = self.fromEulerAngleToMatrix(alpha, beta, gamma )
            boxLen = torch.exp(boxLen )
            axes = axes * boxLen.unsqueeze(-1 )

            x_int = F.relu(self.gnInt(self.convInt_1(x_src ) ), True )
            x_int = self.convInt_2(x_int )
            x_int = torch.sigmoid(x_int ) * 0.999
            x_int = torch.tan(np.pi / 2.0 * x_int )
            x_int = x_int.view(batchSize, 3 )

            lightSrc = x_int

            return axes, center, lightSrc

        else:
            x_center, x_src = torch.split(x, [512, 512], dim=1 )

            x_center = F.relu(self.gnCenter(self.convCenter_1(x_center ) ), True )
            x_center = torch.tanh(self.convCenter_2(x_center )  )
            center = 0.2 * x_center * centerDirec + center
            center = center.view(batchSize, 3 )

            x_int = F.relu(self.gnInt(self.convInt_1(x_src ) ), True )
            x_int = self.convInt_2(x_int )
            x_int = torch.sigmoid(x_int ) * 0.9999
            x_int = torch.tan(np.pi / 2.0 * x_int )
            x_int = x_int.view(batchSize, 3 )

            lightSrc = x_int

            return center, lightSrc


class windowNet(nn.Module ):
    def __init__(self, isInv, fov=57.95 ):
        super(windowNet, self).__init__()
        # Predict the light source parameters of the visible light sources
        self.isInv = isInv
        self.fov = fov / 180.0 * np.pi

        self.encoder = lightSrcEncoder(isInv, isFg = False )

        self.convGeo_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
        self.gnGeo = nn.GroupNorm(num_groups=16, num_channels=256 )
        self.convGeo_2 = nn.Conv2d(in_channels=256, out_channels=11, kernel_size=1, stride=1, bias=True )

        self.convSrcSun_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
        self.gnSrcSun = nn.GroupNorm(num_groups=16, num_channels=256 )
        self.convSrcSun_2 = nn.Conv2d(in_channels=256, out_channels=7, kernel_size=1, stride=1, bias=True )

        self.convSrcSky_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
        self.gnSrcSky = nn.GroupNorm(num_groups=16, num_channels=256 )
        self.convSrcSky_2 = nn.Conv2d(in_channels=256, out_channels=7, kernel_size=1, stride=1, bias=True )

        self.convSrcGrd_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, bias=True )
        self.gnSrcGrd = nn.GroupNorm(num_groups=16, num_channels=256 )
        self.convSrcGrd_2 = nn.Conv2d(in_channels=256, out_channels=7, kernel_size=1, stride=1, bias=True )

    def transformLightSrcPara(self, src, minLamb, maxLamb, initialDirec = None ):
        batchSize = src.size(0 )
        intensity, direc, lamb = torch.split(src, [3, 3, 1], dim=1 )

        intensity = torch.sigmoid(intensity ) *  0.9999999 # 7 9 here
        intensity = torch.tan(np.pi / 2.0 * intensity )
        intensity = intensity.view(batchSize, 3 )

        direc = 2 * torch.tanh(direc )
        if not initialDirec is None:
            direc = direc + initialDirec
        direc = direc.view(batchSize, 3 )

        direc = direc / torch.sqrt(
                torch.clamp(
                    torch.sum(
                        direc * direc,
                        dim=1, keepdim=True
                        ),
                    min = 1e-8
                    )
                )

        lamb = torch.sigmoid(lamb ) * (maxLamb - minLamb ) + minLamb
        lamb = torch.tan(np.pi / 2.0 * lamb )
        lamb = lamb.view(batchSize, 1 )

        lightSrc = torch.cat([intensity, direc, lamb ], dim=1 )

        return lightSrc

    def computeInvCenter(self, centerRaw, height, width ):
        theta, phi, dist = torch.split(centerRaw, [1, 1, 1], dim=1 )
        zAxis = torch.FloatTensor([0, 0, 1] ).cuda().reshape(1, 3)
        yAxis = torch.FloatTensor([0, 1, 0] ).cuda().reshape(1, 3)
        xAxis = torch.FloatTensor([1, 0, 0] ).cuda().reshape(1, 3)

        tanx = np.tan(self.fov / 2.0 )
        tany = tanx / float(width ) * float(height )
        fovy = np.arctan(tany ) * 2.0

        theta = torch.sigmoid(theta ) * (np.pi - fovy )
        phi = torch.tanh(phi ) * np.pi
        dist = torch.exp(dist )

        centerDirec = torch.sin(theta ) * torch.cos(phi ) * xAxis\
            + torch.sin(theta ) * torch.sin(phi ) * yAxis \
            + torch.cos(theta ) * zAxis
        center = centerDirec * dist

        return center

    def forward(self, im, depth, albedo,
            lightOnMasks, lightOnMask = None ):

        batchSize = im.size(0 )
        point = depthToPoint(self.fov, depth )

        if self.isInv:
            inputs = torch.cat([im, point, albedo,
                                lightOnMasks ], dim=1 )
        else:
            cropInputs, centerInit, centerDirec = crop2Dregion(
                lightOnMask, depth, isWin=True, fov = self.fov,
                newWidth = depth.size(3), newHeight = depth.size(2)
            )
            inputs = torch.cat([im, point, albedo,
                lightOnMask, lightOnMasks, cropInputs ], dim=1 )

        upInitial = torch.tensor([0, 1, 0], dtype=torch.float32 ).cuda().view(1, 3, 1, 1)

        x = self.encoder(inputs )

        x_geo, x_src = torch.split(x, [512, 512], dim=1 )

        x_geo = F.relu(self.gnGeo(self.convGeo_1(x_geo ) ), True )
        x_geo = self.convGeo_2(x_geo )

        center, normal, up, axisLen = torch.split(x_geo, [3, 3, 3, 2], dim=1 )
        if self.isInv:
            normal = torch.tanh(normal.view(batchSize, 3 ) )
            normal = normal / torch.sqrt(
                    torch.clamp(
                        torch.sum(
                            normal * normal,
                            dim=1, keepdim=True ),
                        min=1e-8 )
                    )

            center = center.view(batchSize, 3 )
            center = self.computeInvCenter(center, height = depth.size(2), width = depth.size(3) )
            centerDirec = None
        else:
            normal = 0.8 * torch.tanh(normal )
            normal = (normal + centerDirec ).view(batchSize, 3 )
            normal = normal / torch.sqrt(
                    torch.clamp(
                        torch.sum(
                            normal * normal,
                            dim=1, keepdim=True ),
                        min=1e-8 )
                    )
            center = torch.tanh(center ) * 0.5 + centerInit
            center = center.view(batchSize, 3 )

        yAxis = 0.8 * torch.tanh(up ) + upInitial
        yAxis = yAxis.view(batchSize, 3)
        yAxis = yAxis - torch.sum(yAxis * normal, dim=1, keepdim=True) * normal
        yAxis = yAxis / torch.sqrt(
                torch.clamp(
                    torch.sum(
                        yAxis * yAxis,
                        dim=1, keepdim=True ),
                    min=1e-6 )
                )
        xAxis = torch.cross(yAxis, normal, dim=-1 )

        normal_reverse_ind = (torch.sum(center * normal, dim=-1, keepdim = True) < 0).float().detach()
        normal = normal * (1 - normal_reverse_ind ) - normal_reverse_ind * normal

        axisLen = torch.exp(axisLen.view(batchSize, 2 ) )
        xAxisLen, yAxisLen = torch.split(axisLen, [1, 1], dim=1 )
        xAxis = xAxisLen * xAxis
        yAxis = yAxisLen * yAxis

        srcSun = F.relu(self.gnSrcSun(self.convSrcSun_1(x_src ) ), True )
        srcSun = self.convSrcSun_2(srcSun )
        lightSrcSun = self.transformLightSrcPara(srcSun, minLamb = 0.9, maxLamb = 0.999999, # 6 9 here
                initialDirec = centerDirec )

        srcSky = F.relu(self.gnSrcSky(self.convSrcSky_1(x_src ) ), True )
        srcSky = self.convSrcSky_2(srcSky )
        lightSrcSky = self.transformLightSrcPara(srcSky, minLamb = 0, maxLamb = 0.9999, # 4 9 here
                initialDirec = upInitial )

        srcGrd = F.relu(self.gnSrcGrd(self.convSrcGrd_1(x_src ) ), True )
        srcGrd = self.convSrcGrd_2(srcGrd )
        lightSrcGrd = self.transformLightSrcPara(srcGrd, minLamb = 0, maxLamb = 0.9999, # 4 9 here
                initialDirec = -upInitial )

        return center, normal, yAxis, xAxis, lightSrcSun, lightSrcSky, lightSrcGrd


class indirectLightNet(nn.Module ):
    def __init__(self, fov=57.95 ):
        super(indirectLightNet, self).__init__()
        self.fov = fov / 180.0 * np.pi

        # Feature Net
        self.padFeat_1 = nn.ReplicationPad2d(1 )
        self.convFeat_1 = nn.Conv2d(in_channels=10, out_channels=64, kernel_size=3, stride=1 )
        self.gnFeat_1 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padDirect_1 = nn.ReplicationPad2d(1 )
        self.convDirect_1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1 )
        self.gnDirect_1 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padEncode_1 = nn.ReplicationPad2d(1 )
        self.convEncode_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4, stride=2 )
        self.gnEncode_1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padEncode_2 = nn.ReplicationPad2d(1 )
        self.convEncode_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2 )
        self.gnEncode_2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padEncode_3 = nn.ReplicationPad2d(1 )
        self.convEncode_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2 )
        self.gnEncode_3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padEncode_4 = nn.ReplicationPad2d(1 )
        self.convEncode_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2 )
        self.gnEncode_4 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.padEncode_5 = nn.ReplicationPad2d(1 )
        self.convEncode_5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1 )
        self.gnEncode_5 = nn.GroupNorm(num_groups=32, num_channels=512 )


        self.padDecode_3 = nn.ReplicationPad2d(1 )
        self.convDecode_3 = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=3, stride=1 )
        self.gnDecode_3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.padDecode_2 = nn.ReplicationPad2d(1 )
        self.convDecode_2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1 )
        self.gnDecode_2 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padDecode_1 = nn.ReplicationPad2d(1 )
        self.convDecode_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1 )
        self.gnDecode_1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.padDecode_0 = nn.ReplicationPad2d(1 )
        self.convDecode_0 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1 )
        self.gnDecode_0 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.padFinal = nn.ReplicationPad2d(1 )
        self.convFinal = nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1 )

    def forward(self, albedo, normal, depth, directShading, lightOnMasks ):

        # Compute the reflectance
        xreflect = albedo * directShading

        # Compute the light transport kernel
        point = depthToPoint(self.fov, depth  )
        inputFeat = torch.cat([point, normal, albedo, lightOnMasks], dim=1 )

        inputDirect = torch.cat([xreflect, directShading], dim=1 )

        # Compute the feature
        xFeat_1 = F.relu(self.gnFeat_1(self.convFeat_1(self.padFeat_1(inputFeat ) ) ), True )
        xDirec_1 = F.relu(self.gnDirect_1(self.convDirect_1(self.padDirect_1(inputDirect ) ) ), True )

        height, width = xFeat_1.size()[2:]
        x1 = F.adaptive_avg_pool2d(torch.cat([xFeat_1, xDirec_1], dim=1),
                                   [int(height / 2.0), int(width /2.0)] )

        # Compute the feature from light source prediction
        x2 = F.relu(self.gnEncode_1(self.convEncode_1(self.padEncode_1(x1 ) ) ), True )
        x3 = F.relu(self.gnEncode_2(self.convEncode_2(self.padEncode_2(x2 ) ) ), True )
        x4 = F.relu(self.gnEncode_3(self.convEncode_3(self.padEncode_3(x3 ) ) ), True )
        x5 = F.relu(self.gnEncode_4(self.convEncode_4(self.padEncode_4(x4 ) ) ), True )
        x6 = F.relu(self.gnEncode_5(self.convEncode_5(self.padEncode_5(x5 ) ) ), True )

        x6 = F.interpolate(x6, x4.size()[2:], mode='bilinear')
        dx4 = torch.cat([x6, x4], dim=1 )
        dx3 = F.relu(self.gnDecode_3(self.convDecode_3(self.padDecode_3(dx4 ) ) ), True )

        dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3) ], mode='bilinear')
        dx3 = torch.cat([dx3, x3], dim=1 )
        dx2 = F.relu(self.gnDecode_2(self.convDecode_2(self.padDecode_2(dx3 ) ) ), True )

        dx2 = F.interpolate(dx2, [x2.size(2), x2.size(3) ], mode='bilinear')
        dx2 = torch.cat([dx2, x2], dim=1 )
        dx1 = F.relu(self.gnDecode_1(self.convDecode_1(self.padDecode_1(dx2 ) ) ) )

        dx1 = F.interpolate(dx1, [x1.size(2), x1.size(3)], mode='bilinear')
        dx1 = torch.cat([dx1, x1], dim=1 )
        dx0 = F.relu(self.gnDecode_0(self.convDecode_0(self.padDecode_0(dx1 ) ) ) )

        dx0 = F.interpolate(dx0, xFeat_1.size()[2:], mode='bilinear' )
        dx0 = torch.cat([dx0, xFeat_1], dim=1 )
        xFinal = torch.sigmoid(self.convFinal(self.padFinal(dx0 ) ) )

        xFinal = torch.tan(0.99 * xFinal * np.pi / 2.0 )

        return xFinal


class encoderLight(nn.Module ):
    def __init__(self ):
        super(encoderLight, self).__init__()

        self.preProcess = nn.Sequential(
                nn.ReplicationPad2d(1),
                nn.Conv2d(in_channels=9, out_channels=32, kernel_size=3, stride=1, bias =True),
                nn.GroupNorm(num_groups=2, num_channels=32),
                nn.ReLU(inplace = True ),

                nn.ZeroPad2d(1 ),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, bias=True),
                nn.GroupNorm(num_groups=4, num_channels=64 ),
                nn.ReLU(inplace = True )
        )

        self.pad0 = nn.ReplicationPad2d(1)
        self.conv0 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, bias = True )
        self.gn0 = nn.GroupNorm(num_groups=4, num_channels=64 )

        self.pad1 = nn.ReplicationPad2d(1 )
        self.conv1 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4, stride = 2, bias = True )
        self.gn1 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.pad3 = nn.ReplicationPad2d(1)
        self.conv3 = nn.Conv2d(in_channels = 256, out_channels=256, kernel_size=4, stride=2, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=16, num_channels=256)

        self.pad4 = nn.ReplicationPad2d(1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=512)

        self.pad5 = nn.ReplicationPad2d(1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=2, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.pad6 = nn.ReplicationPad2d(1)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, bias=True )
        self.gn6 = nn.GroupNorm(num_groups=64, num_channels=1024 )

    def forward(self, inputBatch, shading ):

        input1 = self.preProcess(inputBatch )
        input2 = F.relu(self.gn0(self.conv0(self.pad0(shading ) ) ), True )

        x = torch.cat([input1, input2 ], dim=1 )
        x1 = F.relu(self.gn1(self.conv1(self.pad1(x) ) ), True )
        x2 = F.relu(self.gn2(self.conv2(self.pad2(x1) ) ), True )
        x3 = F.relu(self.gn3(self.conv3(self.pad3(x2) ) ), True )
        x4 = F.relu(self.gn4(self.conv4(self.pad4(x3) ) ), True )
        x5 = F.relu(self.gn5(self.conv5(self.pad5(x4) ) ), True )
        x6 = F.relu(self.gn6(self.conv6(self.pad6(x5) ) ), True )

        return x1, x2, x3, x4, x5, x6


class decoderLight(nn.Module ):
    def __init__(self, SGNum = 12,  mode = 0):
        super(decoderLight, self).__init__()

        self.SGNum = SGNum

        self.dconv1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn1 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn2 = nn.GroupNorm(num_groups=32, num_channels=512 )

        self.dconv3 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True )
        self.dgn3 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn4 = nn.GroupNorm(num_groups=16, num_channels=256 )

        self.dconv5 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn5 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dconv6 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding = 1, bias=True )
        self.dgn6 = nn.GroupNorm(num_groups=8, num_channels=128 )

        self.dpadFinal = nn.ReplicationPad2d(1)

        if mode == 0 or mode == 2:
            self.dconvFinal = nn.Conv2d(in_channels=128, out_channels = 3*SGNum, kernel_size=3, stride=1, bias=True )
        elif mode == 1:
            self.dconvFinal = nn.Conv2d(in_channels=128, out_channels = SGNum, kernel_size=3, stride=1, bias=True )

        self.mode = mode

    def forward(self, x1, x2, x3, x4, x5, x6, env = None):
        dx1 = F.relu(self.dgn1(self.dconv1(x6 ) ) )

        xin1 = torch.cat([dx1, x5], dim = 1)
        dx2 = F.relu(self.dgn2(self.dconv2(F.interpolate(xin1, scale_factor=2, mode='bilinear') ) ), True)

        if dx2.size(3) != x4.size(3) or dx2.size(2) != x4.size(2):
            dx2 = F.interpolate(dx2, [x4.size(2), x4.size(3)], mode='bilinear')
        xin2 = torch.cat([dx2, x4], dim=1 )
        dx3 = F.relu(self.dgn3(self.dconv3(F.interpolate(xin2, scale_factor=2, mode='bilinear') ) ), True)

        if dx3.size(3) != x3.size(3) or dx3.size(2) != x3.size(2):
            dx3 = F.interpolate(dx3, [x3.size(2), x3.size(3)], mode='bilinear')
        xin3 = torch.cat([dx3, x3], dim=1)
        dx4 = F.relu(self.dgn4(self.dconv4(F.interpolate(xin3, scale_factor=2, mode='bilinear') ) ), True)

        if dx4.size(3) != x2.size(3) or dx4.size(2) != x2.size(2):
            dx4 = F.interpolate(dx4, [x2.size(2), x2.size(3)], mode='bilinear')
        xin4 = torch.cat([dx4, x2], dim=1 )
        dx5 = F.relu(self.dgn5(self.dconv5(F.interpolate(xin4, scale_factor=2, mode='bilinear') ) ), True)

        if dx5.size(3) != x1.size(3) or dx5.size(2) != x1.size(2):
            dx5 = F.interpolate(dx5, [x1.size(2), x1.size(3)], mode='bilinear')
        xin5 = torch.cat([dx5, x1], dim=1 )
        dx6 = F.relu(self.dgn6(self.dconv6(F.interpolate(xin5, scale_factor=2, mode='bilinear') ) ), True)

        if dx6.size(3) != env.size(3) or dx6.size(2) != env.size(2):
            dx6 = F.interpolate(dx6, [env.size(2), env.size(3)], mode='bilinear')
        x_orig = self.dconvFinal(self.dpadFinal(dx6 ) )

        x_out = 1.01 * torch.tanh(self.dconvFinal(self.dpadFinal(dx6) ) )

        if self.mode == 1 or self.mode == 2:
            x_out = 0.5 * (x_out + 1)
            x_out = torch.clamp(x_out, 0, 1 )
        elif self.mode == 0:
            bn, _, row, col = x_out.size()
            x_out = x_out.view(bn, self.SGNum, 3, row, col)
            x_out = x_out / torch.clamp(torch.sqrt(torch.sum(x_out * x_out,
                dim=2).unsqueeze(2) ), min = 1e-6).expand_as(x_out )
        return x_out


