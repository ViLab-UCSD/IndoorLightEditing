import utils
import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import models
import modelLight
import renderWindow
import renderVisLamp
import renderInvLamp
import torchvision.utils as vutils
import torchvision.models as vmodels
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
import lossFunctions
import scipy.ndimage as ndimage
import renderShadowDepth
import pickle
import glob
import cv2


def renderVisWindowArr(
    visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
    depthDS, normalDS,
    visWinShadowPreds ):

    visWinShadingPreds = []
    visWinShadingNoPreds = []

    for n in range(0, len(visWinCenterPreds ) ):
        visWinShadingNoPred \
                = renderWindow.forward(
                visWinCenterPreds[n],
                visWinNormalPreds[n],
                visWinYAxisPreds[n],
                visWinXAxisPreds[n],
                visWinSrcPreds[n],
                visWinSrcSkyPreds[n],
                visWinSrcGrdPreds[n],
                depthDS.detach(),
                normalDS.detach() )

        visWinShadingNoPreds.append(visWinShadingNoPred )
        visWinShadingPreds.append(visWinShadingNoPred * visWinShadowPreds[:, n] )

    return visWinShadingPreds, visWinShadingNoPreds


def renderVisLampArr(
    visLampCenterPreds, visLampSrcPreds, onLampMasksSmallBatch,
    depthDS, normalDS,
    visLampShadowPreds ):

    visLampShadingPreds = []
    visLampShadingNoPreds = []

    for n in range(0, len(visLampCenterPreds ) ):
        visLampShadingNoPred, _ \
                = renderVisLamp.forward(
                visLampCenterPreds[n],
                visLampSrcPreds[n],
                depthDS.detach(),
                    onLampMasksSmallBatch[:, n:n+1, :],
                normalDS.detach(),
                isTest = False )

        visLampShadingNoPreds.append(visLampShadingNoPred )

        visLampShadingNoPredSelfOcclu, visLampPointsPred \
                = renderVisLamp.forward(
                visLampCenterPreds[n],
                visLampSrcPreds[n],
                depthDS.detach(),
                onLampMasksSmallBatch[:, n:n+1, :],
                normalDS.detach(),
                isTest = True )
        visLampShadingPreds.append(visLampShadingNoPredSelfOcclu * visLampShadowPreds[:, n, :] )

    return visLampShadingPreds, visLampShadingNoPreds


def renderInvWindowArr(
    invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
    invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
    depthDS, normalDS,
    invWinShadowPred ):

    invWinShadingNoPred = renderWindow.forward(
                    invWinCenterPred,
                    invWinNormalPred,
                    invWinXAxisPred,
                    invWinYAxisPred,
                    invWinSrcPred,
                    invWinSrcSkyPred,
                    invWinSrcGrdPred,
                    depthDS.detach(),
                    normalDS.detach() )

    invWinShadingPred = invWinShadingNoPred * invWinShadowPred

    return invWinShadingPred, invWinShadingNoPred


def renderInvLampArr(
    invLampAxesPred, invLampCenterPred,
    invLampSrcPred,
    depthDS, normalDS,
    invLampShadowPred ):

    invLampShadingNoPred = renderInvLamp.forward(
                    invLampAxesPred,
                    invLampCenterPred,
                    invLampSrcPred,
                    depthDS.detach(),
                    normalDS.detach(),
                    isTest = False )

    invLampShadingNoPredSelfOcclu = renderInvLamp.forward(
                    invLampAxesPred,
                    invLampCenterPred,
                    invLampSrcPred,
                    depthDS.detach(),
                    normalDS.detach(),
                    isTest = True )
    invLampShadingPred = invLampShadingNoPredSelfOcclu * invLampShadowPred

    return invLampShadingPred, invLampShadingNoPred


def transformParaWin(para, intensity, minLamb, maxLamb ):
    if para is None:
        return None

    color = para[:, 0:3]
    scale = para[:, 3:4]
    direction = para[:, 4:7]
    lamb = para[:, 7:8]

    color = torch.tanh(color ) * 0.01 + torch.ones(1, 3, dtype = torch.float32 ).cuda()
    scale = torch.exp(scale )
    intensity = color * scale * intensity

    direction = F.normalize(direction, dim=1 )
    lamb = torch.sigmoid(lamb ) * (maxLamb - minLamb ) + minLamb
    lamb = torch.tan(np.pi / 2.0 * lamb )

    para = torch.cat([intensity, direction, lamb ], dim=1 )
    return para

def transformParaWinInverse(para, minLamb, maxLamb ):
    if para is None:
        return None

    direction = para[:, 3:6 ]
    lamb = para[:, 6:7 ]

    color = torch.zeros([1, 3], dtype = torch.float32 ).cuda()
    scale = torch.zeros([1, 1], dtype = torch.float32 ).cuda()

    direction = F.normalize(direction, dim=1 )

    lamb = torch.arctan(lamb ) / np.pi  * 2.0
    lamb = torch.clamp(lamb  - minLamb, 1e-12, 1-1e-12 ) / (maxLamb - minLamb )
    lamb = torch.log(lamb ) - torch.log(1 - lamb )

    para = torch.cat([color, scale, direction, lamb], dim=1 )

    return para


def transformParaLamp(para, intensity ):
    if para is None:
        return None

    color = para[:, 0:3]
    scale = para[:, 3:4]

    color = torch.tanh(color ) * 0.01 + torch.ones(1, 3, dtype = torch.float32 ).cuda()
    scale = torch.exp(scale )
    intensity = color * scale * intensity

    return intensity

def transformParaLampInverse(para ):
    if para is None:
        return None

    color = torch.zeros([1, 3], dtype = torch.float32 ).cuda()
    scale = torch.zeros([1, 1], dtype = torch.float32 ).cuda()

    para = torch.cat([color, scale], dim=1 )

    return para

def transformPara(
    visWinSrcPreds_i, visWinSrcSkyPreds_i, visWinSrcGrdPreds_i,
    visWinSrcInts, visWinSrcSkyInts, visWinSrcGrdInts,
    visLampSrcPreds_i,
    visLampSrcInts,
    invWinSrcPred_i, invWinSrcSkyPred_i, invWinSrcGrdPred_i,
    invWinSrcInt, invWinSrcSkyInt, invWinSrcGrdInt,
    invLampSrcPred_i,
    invLampSrcInt ):

    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds = [], [], []
    visLampSrcPreds = []

    for n in range(0, len(visWinSrcPreds_i ) ):
        visWinSrcPreds.append(transformParaWin(visWinSrcPreds_i[n], visWinSrcInts[n],
                                               maxLamb = 0.9999999, minLamb = 0.9 ) )
        visWinSrcSkyPreds.append(transformParaWin(visWinSrcSkyPreds_i[n], visWinSrcSkyInts[n],
                                                  maxLamb = 0.9999, minLamb = 1e-8 ) )
        visWinSrcGrdPreds.append(transformParaWin(visWinSrcGrdPreds_i[n], visWinSrcGrdInts[n],
                                                  maxLamb = 0.9999, minLamb = 1e-8 ) )

    for n in range(0, len(visLampSrcPreds_i ) ):
        visLampSrcPreds.append(transformParaLamp(visLampSrcPreds_i[n], visLampSrcInts[n] ) )

    invWinSrcPred = transformParaWin(invWinSrcPred_i, invWinSrcInt,
                                     maxLamb = 0.9999999, minLamb = 0.9 )
    invWinSrcSkyPred = transformParaWin(invWinSrcSkyPred_i, invWinSrcSkyInt,
                                        maxLamb = 0.9999, minLamb = 1e-8 )
    invWinSrcGrdPred = transformParaWin(invWinSrcGrdPred_i, invWinSrcGrdInt,
                                        maxLamb = 0.9999, minLamb = 1e-8 )

    invLampSrcPred = transformParaLamp(invLampSrcPred_i, invLampSrcInt )

    return visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds, \
        visLampSrcPreds, \
        invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred, \
        invLampSrcPred


def transformParaInverse(
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
    visLampSrcPreds,
    invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
    invLampSrcPred ):

    visWinSrcPreds_i, visWinSrcSkyPreds_i, visWinSrcGrdPreds_i = [], [], []
    visWinSrcInts, visWinSrcSkyInts, visWinSrcGrdInts = [], [], []
    visLampSrcPreds_i = []
    visLampSrcInts = []

    for n in range(0, len(visWinSrcPreds ) ):
        visWinSrcPreds_i.append(transformParaWinInverse(visWinSrcPreds[n], maxLamb = 0.9999999, minLamb = 0.9 ) )
        if visWinSrcPreds[n] is not None:
            visWinSrcInts.append(visWinSrcPreds[n][:, 0:3].detach() )
        else:
            visWinSrcInts.append(None )

        visWinSrcSkyPreds_i.append(transformParaWinInverse(visWinSrcSkyPreds[n], maxLamb = 0.9999, minLamb = 1e-8 ) )
        if visWinSrcSkyPreds[n] is not None:
            visWinSrcSkyInts.append(visWinSrcSkyPreds[n][:, 0:3].detach() )
        else:
            visWinSrcSkyInts.append(None )

        visWinSrcGrdPreds_i.append(transformParaWinInverse(visWinSrcGrdPreds[n], maxLamb = 0.9999, minLamb = 1e-8 ) )
        if visWinSrcGrdPreds[n] is not None:
            visWinSrcGrdInts.append(visWinSrcGrdPreds[n][:, 0:3].detach() )
        else:
            visWinSrcGrdInts.append(None )

    for n in range(0, len(visLampSrcPreds ) ):
        visLampSrcPreds_i.append(transformParaLampInverse(visLampSrcPreds[n] ) )
        if visLampSrcPreds[n] is not None:
            visLampSrcInts.append(visLampSrcPreds[n][:, 0:3].detach() )
        else:
            visLampSrcInts.append(None )

    invWinSrcPred_i = transformParaWinInverse(invWinSrcPred, maxLamb = 0.9999999, minLamb = 0.9 )
    invWinSrcInt = invWinSrcPred[:, 0:3]

    invWinSrcSkyPred_i = transformParaWinInverse(invWinSrcSkyPred, maxLamb = 0.9999, minLamb = 1e-8 )
    invWinSrcSkyInt = invWinSrcSkyPred[:, 0:3]

    invWinSrcGrdPred_i = transformParaWinInverse(invWinSrcGrdPred, maxLamb = 0.9999, minLamb = 1e-8 )
    invWinSrcGrdInt = invWinSrcGrdPred[:, 0:3]

    invLampSrcPred_i = transformParaLampInverse(invLampSrcPred )
    invLampSrcInt = invLampSrcPred[:, 0:3]

    return visWinSrcPreds_i, visWinSrcInts, \
        visWinSrcSkyPreds_i, visWinSrcSkyInts, \
        visWinSrcGrdPreds_i, visWinSrcGrdInts, \
        visLampSrcPreds_i, visLampSrcInts, \
        invWinSrcPred_i,  invWinSrcInt, \
        invWinSrcSkyPred_i, invWinSrcSkyInt, \
        invWinSrcGrdPred_i, invWinSrcGrdInt, \
        invLampSrcPred_i, invLampSrcInt

def optimizeLightSources(
    # Visible window parameters
    visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
    onWinMasksSmallBatch,
    visWinShadowPreds,
    # Visible lamp parameters
    visLampCenterPreds, visLampSrcPreds,
    onLampMasksSmallBatch,
    visLampShadowPreds,
    # Invisible window parameters
    invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
    invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
    invWinShadowPred,
    # Invisible lamp parameters
    invLampAxesPred, invLampCenterPred,
    invLampSrcPred,
    invLampShadowPred,
    # Image
    imSmallBatch,
    depthDS, normalDS, albedoDS,
    envMaskSmallBatch,
    onMasksSmallBatch,
    iterNum ):

    height, width = imSmallBatch.size()[2:]
    ignoreMaskOrig = torch.zeros(1, 1, height, width, dtype = torch.float32 ).cuda()
    if len(visWinCenterPreds ) > 0:
        ignoreMaskOrig += torch.sum(onWinMasksSmallBatch, dim=1, keepdim=True)
    if len(visLampCenterPreds ) > 0:
        ignoreMaskOrig += torch.sum(onLampMasksSmallBatch, dim=1, keepdim=True )

    visWinNum = len(visWinCenterPreds )
    visLampNum = len(visLampCenterPreds )

    saturatedMaskBatch = (torch.mean(imSmallBatch, dim=1, keepdim=True) == 1).float()

    ignoreMask = ignoreMaskOrig[0, 0, :]
    ignoreMask = (ignoreMask.detach().cpu().numpy() == 1)

    ignoreMask = ndimage.binary_dilation(ignoreMask, structure = np.ones((7, 7) ) )

    saturatedMask = (saturatedMaskBatch[0, 0, :].detach().cpu().numpy() == 1)

    ignoreMask = (ndimage.binary_dilation(saturatedMask, structure = np.ones((7, 7) ) ) \
                    ^ saturatedMask | ignoreMask  )[np.newaxis, np.newaxis, : ]

    ignoreMask = ignoreMask.astype(np.float32 )

    ignoreMaskBatch = torch.from_numpy(ignoreMask ).cuda()

    ignoreMaskBatch = 1 - ignoreMaskBatch

    visWinSrcPreds_i, visWinSrcInts, \
        visWinSrcSkyPreds_i, visWinSrcSkyInts, \
        visWinSrcGrdPreds_i, visWinSrcGrdInts, \
        visLampSrcPreds_i, visLampSrcInts, \
        invWinSrcPred_i, invWinSrcInt, \
        invWinSrcSkyPred_i, invWinSrcSkyInt, \
        invWinSrcGrdPred_i, invWinSrcGrdInt, \
        invLampSrcPred_i, invLampSrcInt = \
        transformParaInverse(
            visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
            visLampSrcPreds,
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
            invLampSrcPred )

    lightSrcParas_i = []
    for visWinSrcPred_i in visWinSrcPreds_i:
        if visWinSrcPred_i is not None:
            visWinSrcPred_i.requires_grad = True
            lightSrcParas_i.append(visWinSrcPred_i )
    for visWinSrcSkyPred_i in visWinSrcSkyPreds_i:
        if visWinSrcSkyPred_i is not None:
            visWinSrcSkyPred_i.requires_grad = True
            lightSrcParas_i.append(visWinSrcSkyPred_i )
    for visWinSrcGrdPred_i in visWinSrcGrdPreds_i:
        if visWinSrcGrdPred_i is not None:
            visWinSrcGrdPred_i.requires_grad = True
            lightSrcParas_i.append(visWinSrcGrdPred_i )
    for visLampSrcPred_i in visLampSrcPreds_i:
        if visLampSrcPred_i is not None:
            visLampSrcPred_i.requires_grad = True
            lightSrcParas_i.append(visLampSrcPred_i )

    invWinSrcPred_i.requires_grad = True
    invWinSrcSkyPred_i.requires_grad = True
    invWinSrcGrdPred_i.requires_grad = True
    invLampSrcPred_i.requires_grad = True

    lightSrcParas_i += [invWinSrcPred_i, invWinSrcSkyPred_i, invWinSrcGrdPred_i ]  \
        + [invLampSrcPred_i ]
    opLightSources = optim.Adam(lightSrcParas_i, lr=1e-1, betas=(0.9, 0.999 ) )

    visWinSrcPreds_b, visWinSrcSkyPreds_b, visWinSrcGrdPreds_b = [], [], []
    for n in range(0, len(visWinSrcPreds ) ):
        visWinSrcPreds_b.append(None )
        visWinSrcSkyPreds_b.append(None )
        visWinSrcGrdPreds_b.append(None )
    visLampSrcPreds_b = []
    for n in range(0, len(visLampSrcPreds ) ):
        visLampSrcPreds_b.append(None )
    invWinSrcPred_b, invWinSrcSkyPred_b, invWinSrcGrdPred_b = None, None, None
    invLampSrcPred_b = None

    minLoss = 1e10
    lossSum = 0

    for n in range(0,  iterNum ):
        opLightSources.zero_grad()

        visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds, \
            visLampSrcPreds, \
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred, \
            invLampSrcPred \
            = transformPara(
                visWinSrcPreds_i, visWinSrcSkyPreds_i, visWinSrcGrdPreds_i,
                visWinSrcInts, visWinSrcSkyInts, visWinSrcGrdInts,
                visLampSrcPreds_i,
                visLampSrcInts,
                invWinSrcPred_i, invWinSrcSkyPred_i, invWinSrcGrdPred_i,
                invWinSrcInt, invWinSrcSkyInt, invWinSrcGrdInt,
                invLampSrcPred_i,
                invLampSrcInt
            )

        # Compute rendering error for visible window
        visWinShadingPreds, visWinShadingNoPreds, \
            = renderVisWindowArr(
                # Visible window parameters
                visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
                visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
                depthDS, normalDS,
                visWinShadowPreds
            )

        # Compute rendering error for visible lamp
        visLampShadingPreds, visLampShadingNoPreds, \
            = renderVisLampArr(
                visLampCenterPreds, visLampSrcPreds, onLampMasksSmallBatch,
                depthDS, normalDS,
                visLampShadowPreds
            )

        # Compute rendering error for invisible window
        invWinShadingPred, invWinShadingNoPred, \
            = renderInvWindowArr(
                invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
                invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
                depthDS, normalDS,
                invWinShadowPred
            )

        # Compute rendering error for invisible lamp
        invLampShadingPred, invLampShadingNoPred, \
            = renderInvLampArr(
                invLampAxesPred, invLampCenterPred,
                invLampSrcPred,
                depthDS, normalDS,
                invLampShadowPred
            )

        # Predict the global illumination
        if len(visWinSrcPreds ) > 0:
            visWinShadingNoPreds = torch.cat(visWinShadingNoPreds, dim=0 ).reshape( 1, visWinNum, 3, opt.envRow, opt.envCol )
            visWinShadingPreds = torch.cat(visWinShadingPreds, dim=0 ).reshape( 1, visWinNum, 3, opt.envRow, opt.envCol )

        if len(visLampSrcPreds ) > 0:
            visLampShadingNoPreds = torch.cat(visLampShadingNoPreds, dim=0 ).reshape(1, visLampNum, 3, opt.envRow, opt.envCol )
            visLampShadingPreds = torch.cat(visLampShadingPreds, dim=0 ).reshape(1, visLampNum, 3, opt.envRow, opt.envCol )

        shadingDirectPred = invWinShadingPred + invLampShadingPred
        if len(visWinSrcPreds ) > 0:
            shadingDirectPred += torch.sum(visWinShadingPreds, dim=1 )

        if len(visLampSrcPreds ) > 0:
            shadingDirectPred += torch.sum(visLampShadingPreds, dim=1 )

        shadingDirectPred = shadingDirectPred * envMaskSmallBatch + (1 - envMaskSmallBatch) * imSmallBatch

        shadingDirectPredInput = torch.atan(shadingDirectPred ) / np.pi * 2.0
        shadingIndirectPred = indirectLightNet(
                albedoDS.detach(),
                normalDS.detach(),
                depthDS.detach(),
                shadingDirectPredInput.detach(),
                onMasksSmallBatch )

        shadingPred = shadingIndirectPred + shadingDirectPred
        renderedPred = torch.clamp(shadingPred * albedoDS, 0, 1 )

        loss = torch.mean( torch.pow(renderedPred - imSmallBatch, 2 ) *  ignoreMaskBatch  )
        lossSum += loss.detach().cpu().numpy()

        loss.backward()
        opLightSources.step()
        print('%d: %.5f' % (n, loss.cpu().detach().numpy() ) )

        if (n+1) % 40 == 0 or n == 0:
            if n != 0:
                lossSum  = lossSum / 40
            if minLoss - lossSum > 0.0001:
                print('Loss decrease: %.4f' % (minLoss - lossSum ) )
                minLoss = lossSum

                for n in range(0, len(visWinSrcPreds ) ):
                    if visWinSrcPreds[n] is not None:
                        visWinSrcPreds_b[n] = visWinSrcPreds[n].detach()
                        visWinSrcSkyPreds_b[n] = visWinSrcSkyPreds[n].detach()
                        visWinSrcGrdPreds_b[n] = visWinSrcGrdPreds[n].detach()

                for n in range(0, len(visLampSrcPreds ) ):
                    if visLampSrcPreds[n] is not None:
                        visLampSrcPreds_b[n] = visLampSrcPreds[n].detach()

                invWinSrcPred_b = invWinSrcPred.detach()
                invWinSrcSkyPred_b = invWinSrcSkyPred.detach()
                invWinSrcGrdPred_b = invWinSrcGrdPred.detach()

                invLampSrcPred_b = invLampSrcPred.detach()
                if n!= 0:
                    lossSum = 0
            else:
                break

        del visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds, \
            visLampSrcPreds, \
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred, \
            invLampSrcPred

    return visWinSrcPreds_b, visWinSrcSkyPreds_b, visWinSrcGrdPreds_b, \
        visLampSrcPreds_b, \
        invWinSrcPred_b, invWinSrcSkyPred_b, invWinSrcGrdPred_b, \
        invLampSrcPred_b, \
        ignoreMaskBatch

parser = argparse.ArgumentParser()
# The directory of trained models
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentDirecIndirec', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentShadow', default=None, help='the path to store samples and models')
parser.add_argument('--testList', default=None, help='the path to store samples and models' )

# The basic training setting
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Training epochs and iterations
parser.add_argument('--nepochBRDF', type=int, default=15, help='the epoch used for testing' )
parser.add_argument('--iterIdVisLamp', type=int, default=119540, help='the iteration used for testing' )
parser.add_argument('--iterIdInvLamp', type=int, default=150000, help='the iteration used for testing' )
parser.add_argument('--iterIdVisWin', type=int, default=120000, help='the iteration used for testing' )
parser.add_argument('--iterIdInvWin', type=int, default=200000, help='the iteration used for testing' )
parser.add_argument('--iterIdDirecIndirec', type=int, default=180000, help='the iteration used for testing' )
parser.add_argument('--iterIdShadow', type=int, default=70000, help='the iteration used for testing')

parser.add_argument('--isOptimize', action='store_true', help='use optimization for light sources or not' )
parser.add_argument('--iterNum', type=int, default = 400, help='the number of interations for optimization')

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for size error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--winSrcIntWeight', type=float, default=0.001, help='the loss for window light source' )
parser.add_argument('--winSrcAxisWeight', type=float, default=1.0, help='the loss for window light source' )
parser.add_argument('--winSrcLambWeight', type=float, default=0.001, help='the loss for window light source' )

# Starting and ending point
parser.add_argument('--rs', type=int, default=0, help='starting point' )
parser.add_argument('--re', type=int, default=1, help='ending point' )


# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

if opt.experimentVisWindow is None:
    opt.experimentVisWindow = 'check_visWindow'
    opt.experimentVisWindow += '_bn3_shg%.3f_geo%.3f_size%.3f_ren%.3f_srcInt%.3f_srcAxis%.3f_srcLamb%.3f' \
        % (opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight,
           opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight )
opt.experimentVisWindow = osp.join(curDir, opt.experimentVisWindow )

if opt.experimentVisLamp is None:
    opt.experimentVisLamp = 'check_visLamp'
    opt.experimentVisLamp += '_bn3_shg%.3f_geo%.3f_ren%.3f' \
        % (opt.shadingWeight, opt.geometryWeight, opt.renderWeight )
opt.experimentVisLamp = osp.join(curDir, opt.experimentVisLamp )

if opt.experimentInvWindow is None:
    opt.experimentInvWindow = 'check_invWindow'
    opt.experimentInvWindow += '_bn3_shg%.3f_geo%.3f_size%.3f_ren%.3f_srcInt%.3f_srcAxis%.3f_arcLamb%.3f' \
        % (opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight,
           opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight )
opt.experimentInvWindow = osp.join(curDir, opt.experimentInvWindow )

if opt.experimentInvLamp is None:
    opt.experimentInvLamp = 'check_invLamp'
    opt.experimentInvLamp += '_bn3_shg%.3f_geo%.3f_size%.3f_ren%.3f' \
        % (opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight )
opt.experimentInvLamp = osp.join(curDir, opt.experimentInvLamp )

if opt.experimentDirecIndirec is None:
    opt.experimentDirecIndirec = 'check_directIndirect'
opt.experimentDirecIndirec = osp.join(curDir, opt.experimentDirecIndirec )

if opt.experimentShadow is None:
    opt.experimentShadow = 'check_shadowDepth'
    opt.experimentShadow += '_grad'
opt.experimentShadow = osp.join(curDir, opt.experimentShadow )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

# Network for lighting prediction
encoder.load_state_dict(torch.load('{0}/encoder_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in encoder.parameters():
    param.requires_grad = False
albedoDecoder.load_state_dict(torch.load('{0}/albedo_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in albedoDecoder.parameters():
    param.requires_grad = False
normalDecoder.load_state_dict(torch.load('{0}/normal_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in normalDecoder.parameters():
    param.requires_grad = False
roughDecoder.load_state_dict(torch.load('{0}/rough_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in roughDecoder.parameters():
    param.requires_grad = False

visWinNet = modelLight.windowNet(isInv = False )
visWinDict = torch.load('{0}/visWinNet_iter{1}.pth'.format(opt.experimentVisWindow, opt.iterIdVisWin ) )
visWinNet.load_state_dict(visWinDict['model'] )
for param in visWinNet.parameters():
    param.requires_grad = False

visLampNet = modelLight.lampNet(isInv = False )
visLampDict = torch.load('{0}/visLampNet_iter{1}.pth'.format(opt.experimentVisLamp, opt.iterIdVisLamp ) )
visLampNet.load_state_dict(visLampDict['model'] )
for param in visLampNet.parameters():
    param.requires_grad = False

invWinNet = modelLight.windowNet(isInv = True )
invWinDict = torch.load('{0}/invWinNet_iter{1}.pth'.format(opt.experimentInvWindow, opt.iterIdInvWin ) )
invWinNet.load_state_dict(invWinDict['model'] )
for param in invWinNet.parameters():
    param.requires_grad = False

invLampNet = modelLight.lampNet(isInv = True )
invLampDict = torch.load('{0}/invLampNet_iter{1}.pth'.format(opt.experimentInvLamp, opt.iterIdInvLamp ) )
invLampNet.load_state_dict(invLampDict['model'] )
for param in invLampNet.parameters():
    param.requires_grad = False

renderWindow = renderWindow.renderDirecLighting(sampleNum = 400 )
renderVisLamp = renderVisLamp.renderDirecLighting()
renderInvLamp = renderInvLamp.renderDirecLighting()
renderShadow = renderShadowDepth.renderShadow(
    modelRoot = opt.experimentShadow, iterId = opt.iterIdShadow,
    winSampleNum = 1024, lampSampleNum = 1024
)

# Network for direct-indirect lighting predictio
indirectLightNet = modelLight.indirectLightNet()
indirectLightDict = torch.load('{0}/indirectLightNet_iter{1}.pth'.format(opt.experimentDirecIndirec, opt.iterIdDirecIndirec ) )
indirectLightNet.load_state_dict(indirectLightDict['model'] )

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
roughDecoder = roughDecoder.cuda()
normalDecoder = normalDecoder.cuda()

visWinNet = visWinNet.cuda()
visLampNet = visLampNet.cuda()
invWinNet = invWinNet.cuda()
invLampNet = invLampNet.cuda()

indirectLightNet = indirectLightNet.cuda()


with open(opt.testList, 'r') as fIn:
    dirList = fIn.readlines()
dirList = [x.strip() for x in dirList if x[0] != '#' ]

timestart = torch.cuda.Event(enable_timing = True )
timestop = torch.cuda.Event(enable_timing = True )


for dataId in range(max(opt.rs, 0), min(opt.re, len(dirList ) ) ):
    dataDir = dirList[dataId ]
    print(dataDir )
    # Load image, assume the the longest width will be 320/1600
    inputDir = osp.join(dataDir, 'input')
    outputDir = osp.join(dataDir, 'BRDFLight')
    outputDir += '_size%.3f_int%.3f_dir%.3f_lam%.3f_ren%.3f' \
        % (opt.sizeWeight, opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight,
           opt.renderWeight )
    outputDir += '_visWin%d_visLamp%d_invWin%d_invLamp%d' \
        % (opt.iterIdVisWin, opt.iterIdVisLamp, opt.iterIdInvWin, opt.iterIdInvLamp )
    if opt.isOptimize:
        outputDir += '_optimize'

    if not osp.isdir(outputDir ):
        os.system('mkdir %s' % outputDir )

    imName = osp.join(inputDir, 'im.png')
    depthName = osp.join(inputDir, 'depth.npy')
    envMaskName = osp.join(inputDir, 'envMask.png')
    lampMaskNames = glob.glob(osp.join(inputDir, 'lampMask_*.png') )
    if len(lampMaskNames ) > 1:
        lampMaskNames = sorted(lampMaskNames )

    winMaskNames = glob.glob(osp.join(inputDir, 'winMask_*.png') )
    if len(winMaskNames ) > 1:
        winMaskNames = sorted(winMaskNames )

    im = cv2.imread(imName )[:, :, ::-1 ]
    originHeight, originWidth = im.shape[0:2 ]
    width = opt.imWidth
    height = int(float(originWidth) / float(width) * originHeight )
    if width != originWidth:
        im = cv2.resize(im, (width, height ), interpolation = cv2.INTER_AREA )
    sWidth, sHeight = int(width / 2.0), int(height / 2.0 )
    imSmall = cv2.resize(im, (sWidth, sHeight), interpolation = cv2.INTER_AREA )

    # depth size should be height x width
    depth = np.load(depthName )

    if width != originWidth:
        depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_AREA )
    depthSmall = cv2.resize(depth, (sWidth, sHeight), interpolation = cv2.INTER_AREA )

    envMask = cv2.imread(envMaskName )
    if width != originWidth:
        envMask = cv2.resize(envMask, (width, height), interpolation = cv2.INTER_AREA )
    envMaskSmall = cv2.resize(envMask, (sWidth, sHeight), interpolation = cv2.INTER_AREA )
    if len(envMask.shape ) == 3:
        envMask = envMask[:, :, 0]
        envMaskSmall = envMaskSmall[:, :, 0]

    lampMasks, lampMaskSmalls = [], []
    for lampMaskName in lampMaskNames:
        lampMask = cv2.imread(lampMaskName )
        if width != originWidth:
            lampMask = cv2.resize(lampMask, (width, height), interpolation = cv2.INTER_AREA )
        lampMaskSmall = cv2.resize(lampMask, (sWidth, sHeight), interpolation = cv2.INTER_AREA )
        if len(lampMask.shape ) == 3:
            lampMask = lampMask[:, :, 0]
            lampMaskSmall = lampMaskSmall[:, :, 0]
        lampMasks.append(lampMask )
        lampMaskSmalls.append(lampMaskSmall )

    winMasks, winMaskSmalls = [], []
    for winMaskName in winMaskNames:
        winMask = cv2.imread(winMaskName )
        if width != originWidth:
            winMask = cv2.resize(winMask, (width, height), interpolation = cv2.INTER_AREA )
        winMaskSmall = cv2.resize(winMask, (sWidth, sHeight), interpolation = cv2.INTER_AREA )
        if len(winMask.shape ) == 3:
            winMask = winMask[:, :, 0]
            winMaskSmall = winMaskSmall[:, :, 0]
        winMasks.append(winMask )
        winMaskSmalls.append(winMaskSmall )

    imBatch = im.transpose(2, 0, 1)[np.newaxis, :, :].astype(np.float32 ) / 255.0
    imBatch = torch.from_numpy(imBatch ** (2.2 )  ).cuda()
    imSmallBatch = imSmall.transpose(2, 0, 1)[np.newaxis, :, :].astype(np.float32 ) / 255.0
    imSmallBatch = torch.from_numpy( imSmallBatch ** (2.2 ) ).cuda()

    depthBatch = depth[np.newaxis, np.newaxis, :, :].astype(np.float32 )
    depthBatch = torch.from_numpy(depthBatch ).cuda()
    depthSmallBatch = depthSmall[np.newaxis, np.newaxis, :, :].astype(np.float32 )
    depthSmallBatch = torch.from_numpy(depthSmallBatch ).cuda()

    envMaskBatch = envMask[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
    envMaskBatch = torch.from_numpy(envMaskBatch ).cuda()
    envMaskSmallBatch = envMaskSmall[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
    envMaskSmallBatch = torch.from_numpy(envMaskSmallBatch ).cuda()

    lampMaskBatch = []
    for lampMask in  lampMasks:
        mask = lampMask[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
        mask = torch.from_numpy(mask ).cuda()
        lampMaskBatch.append(mask )
    if len(lampMaskBatch ) > 0:
        lampMaskBatch = torch.cat(lampMaskBatch, dim=1 )
    lampMaskSmallBatch = []
    for lampMaskSmall in  lampMaskSmalls:
        mask = lampMaskSmall[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
        mask = torch.from_numpy(mask ).cuda()
        lampMaskSmallBatch.append(mask )
    if len(lampMaskSmallBatch ) > 0:
        lampMaskSmallBatch = torch.cat(lampMaskSmallBatch, dim=1 )

    winMaskBatch = []
    for winMask in  winMasks:
        mask = winMask[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
        mask = torch.from_numpy(mask ).cuda()
        winMaskBatch.append(mask )
    if len(winMaskBatch ) > 0:
        winMaskBatch = torch.cat(winMaskBatch, dim=1 )
    winMaskSmallBatch = []
    for winMaskSmall in  winMaskSmalls:
        mask = winMaskSmall[np.newaxis, np.newaxis, :, :].astype(np.float32 ) / 255.0
        mask = torch.from_numpy(mask ).cuda()
        winMaskSmallBatch.append(mask )
    if len(winMaskSmallBatch ) > 0:
        winMaskSmallBatch = torch.cat(winMaskSmallBatch, dim=1 )

    onMaskBatch = torch.zeros([1, 1, height, width], dtype = torch.float32 ).cuda()
    if len(lampMaskBatch ) > 0:
        onMaskBatch += torch.sum(lampMaskBatch, dim=1, keepdim=True )
    if len(winMaskBatch ) > 0:
        onMaskBatch += torch.sum(winMaskBatch, dim=1, keepdim = True )
    onMaskBatch = torch.clamp(onMaskBatch, 0, 1 )

    onMaskSmallBatch = torch.zeros([1, 1, sHeight, sWidth], dtype = torch.float32 ).cuda()
    if len(lampMaskBatch ) > 0:
        onMaskSmallBatch += torch.sum(lampMaskSmallBatch, dim=1, keepdim=True )
    if len(winMaskBatch ) > 0:
        onMaskSmallBatch += torch.sum(winMaskSmallBatch, dim=1, keepdim = True )
    onMaskSmallBatch = torch.clamp(onMaskSmallBatch, 0, 1 )

    inputBatch = torch.cat([imBatch, depthBatch ], dim=1 )

    # Predict the large BRDF
    timestart.record()
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred, af = albedoDecoder(x1, x2, x3,
            x4, x5, x6, [height, width ] )
    normalPred, nf = normalDecoder(x1, x2, x3,
            x4, x5, x6, [height, width] )
    roughPred, rf = roughDecoder(x1, x2, x3,
            x4, x5, x6, [height, width] )
    depthPred = depthBatch

    # Down sample the BRDF
    albedoDS = F.adaptive_avg_pool2d(albedoPred, (sHeight, sWidth ) )
    normalDS = F.adaptive_avg_pool2d(normalPred, (sHeight, sWidth ) )
    roughDS = F.adaptive_avg_pool2d(roughPred, (sHeight, sWidth ) )
    depthDS = depthSmallBatch

    timestop.record()
    torch.cuda.synchronize()
    print('BRDF time: %.3f ms' % timestart.elapsed_time(timestop ) )

    # Save geometry
    utils.writeDepthAsPointClouds(
        depthSmallBatch,
        normalDS,
        envMaskSmallBatch,
        osp.join(outputDir, 'room.ply')
    )

    # Output light source predictions
    timestart.record()
    visLampNum = len(lampMaskNames )
    visLampSrcPreds, visLampCenterPreds = [], []
    for n in range(0, visLampNum ):
        lampName = lampMaskNames[n].replace(inputDir, outputDir )
        lampName = lampName.replace('Mask', 'Src').replace('.png', '.dat')
        lightMask = lampMaskSmallBatch[:, n:n+1, :]

        visLampCenterPred, visLampSrcPred \
                = visLampNet(
                        imSmallBatch,
                        depthDS, albedoDS,
                        onMaskSmallBatch,
                        lightMask
                )
        visLampCenterPreds.append(visLampCenterPred )
        visLampSrcPreds.append(visLampSrcPred )

    timestop.record()
    torch.cuda.synchronize()
    print('Visible lamp time: %.3f ms' % timestart.elapsed_time(timestop ) )

    if visLampNum > 0:
        utils.writeLampList(
                visLampCenterPreds,
                depthDS,
                normalDS,
                lampMaskSmallBatch,
                visLampNum,
                osp.join(outputDir, 'visLampPred.ply' )
        )


    timestart.record()
    visWinNum = len(winMaskNames )
    visWinCenterPreds = []
    visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds = [], [], []
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds = [], [], []
    for n in range(0, visWinNum ):
        winName = winMaskNames[n].replace(inputDir, outputDir )
        winName = winName.replace('Mask', 'Src').replace('.png', '.dat')
        lightMask = winMaskSmallBatch[:, n:n+1, :]

        visWinCenterPred, visWinNormalPred, \
                visWinYAxisPred, visWinXAxisPred, \
                visWinSrcPred, visWinSrcSkyPred, visWinSrcGrdPred \
                = visWinNet(
                        imSmallBatch,
                        depthDS, albedoDS,
                        onMaskSmallBatch,
                        lightMask
                )
        visWinCenterPreds.append(visWinCenterPred )
        visWinXAxisPreds.append(visWinXAxisPred )
        visWinYAxisPreds.append(visWinYAxisPred )
        visWinNormalPreds.append(visWinNormalPred )

        visWinSrcPreds.append(visWinSrcPred )
        visWinSrcSkyPreds.append(visWinSrcSkyPred )
        visWinSrcGrdPreds.append(visWinSrcGrdPred )

    timestop.record()
    torch.cuda.synchronize()
    print('Visible window time: %.3f ms' % timestart.elapsed_time(timestop ) )

    if visWinNum > 0:
        utils.writeWindowList(
            visWinCenterPreds,
            visWinYAxisPreds,
            visWinXAxisPreds,
            visWinNum,
            osp.join(outputDir, 'visWinPred.obj')
        )

    timestart.record()
    invLampAxesPred, invLampCenterPred, invLampSrcPred \
        = invLampNet(
                imSmallBatch,
                depthDS, albedoDS,
                onMaskSmallBatch
        )

    timestop.record()
    torch.cuda.synchronize()
    print('Invisible lamp time: %.3f ms' % timestart.elapsed_time(timestop ) )


    utils.writeLampBatch(
            invLampAxesPred.unsqueeze(1),
            invLampCenterPred.unsqueeze(1),
            np.ones((1, 1) ),
            1,
            osp.join(outputDir, 'invLampPred.ply' )
    )

    timestart.record()
    invWinCenterPred, invWinNormalPred, \
        invWinYAxisPred, invWinXAxisPred, \
        invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred \
        = invWinNet(
                imSmallBatch,
                depthDS, albedoDS,
                onMaskSmallBatch
        )

    timestop.record()
    torch.cuda.synchronize()
    print('Invisible window time: %.3f ms' % timestart.elapsed_time(timestop ) )

    utils.writeWindowBatch(
            invWinCenterPred.unsqueeze(1),
            invWinYAxisPred.unsqueeze(1),
            invWinXAxisPred.unsqueeze(1),
            np.ones( (1, 1 ) ),
            1,
            osp.join(outputDir, 'invWinPred.obj')
    )

    renderShadow.setWinNum(visWinNum )
    renderShadow.setLampNum(visLampNum )

    # Compute shadows
    timestart.record()
    visWinShadowInits, visWinShadowPreds, \
        visLampShadowInits, visLampShadowPreds, \
        invWinShadowInit, invWinShadowPred, \
        invLampShadowInit, invLampShadowPred \
        = renderShadow.forward(
            depthDS, normalDS,
            visWinCenterPreds, visWinXAxisPreds, visWinYAxisPreds, winMaskSmallBatch,
            visLampCenterPreds, lampMaskSmallBatch,
            invWinCenterPred, invWinXAxisPred, invWinYAxisPred,
            invLampAxesPred, invLampCenterPred,
            objName=None )
    timestop.record()
    torch.cuda.synchronize()
    print('Shadow time: %.3f ms' % timestart.elapsed_time(timestop ) )


    if opt.isOptimize:
        visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds, \
            visLampSrcPreds, \
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred, \
            invLampSrcPred, \
            ignoreMaskBatch \
            = optimizeLightSources(
                # Visible window parameters
                visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
                visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
                winMaskSmallBatch,
                visWinShadowPreds,
                # Visible lamp parameters
                visLampCenterPreds, visLampSrcPreds,
                lampMaskSmallBatch,
                visLampShadowPreds,
                # Invisible window parameters
                invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
                invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
                invWinShadowPred,
                # Invisible lamp parameters
                invLampAxesPred, invLampCenterPred,
                invLampSrcPred,
                invLampShadowPred,
                # Image
                imSmallBatch,
                depthDS, normalDS, albedoDS,
                envMaskSmallBatch,
                onMaskSmallBatch,
                opt.iterNum )


    # Compute rendering error for visible window
    timestart.record()
    visWinShadingPreds, visWinShadingNoPreds, \
        = renderVisWindowArr(
            # Visible window parameters
            visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
            visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
            depthDS, normalDS,
            visWinShadowPreds
        )

    # Compute rendering error for visible lamp
    visLampShadingPreds, visLampShadingNoPreds, \
        = renderVisLampArr(
            visLampCenterPreds, visLampSrcPreds, lampMaskSmallBatch,
            depthDS, normalDS,
            visLampShadowPreds
        )

    # Compute rendering error for invisible window
    invWinShadingPred, invWinShadingNoPred, \
        = renderInvWindowArr(
            invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
            depthDS, normalDS,
            invWinShadowPred
        )

    # Compute rendering error for invisible lamp
    invLampShadingPred, invLampShadingNoPred, \
        = renderInvLampArr(
            invLampAxesPred, invLampCenterPred,
            invLampSrcPred,
            depthDS, normalDS,
            invLampShadowPred
        )
    timestop.record()
    torch.cuda.synchronize()
    print('Direct time: %.3f ms' % timestart.elapsed_time(timestop ) )

    # Predict the global illumination
    if len(visWinSrcPreds ) > 0:
        visWinShadingNoPreds = torch.cat(visWinShadingNoPreds, dim=0 ).reshape( 1, visWinNum, 3, opt.envRow, opt.envCol )
        visWinShadingPreds = torch.cat(visWinShadingPreds, dim=0 ).reshape( 1, visWinNum, 3, opt.envRow, opt.envCol )

    if len(visLampSrcPreds ) > 0:
        visLampShadingNoPreds = torch.cat(visLampShadingNoPreds, dim=0 ).reshape(1, visLampNum, 3, opt.envRow, opt.envCol )
        visLampShadingPreds = torch.cat(visLampShadingPreds, dim=0 ).reshape(1, visLampNum, 3, opt.envRow, opt.envCol )

    shadingDirectPred = invWinShadingPred + invLampShadingPred
    if len(visWinSrcPreds ) > 0:
        shadingDirectPred += torch.sum(visWinShadingPreds, dim=1 )

    if len(visLampSrcPreds ) > 0:
        shadingDirectPred += torch.sum(visLampShadingPreds, dim=1 )

    timestart.record()
    shadingDirectPred = shadingDirectPred * envMaskSmallBatch + (1 - envMaskSmallBatch) * imSmallBatch
    shadingDirectPredInput = torch.atan(shadingDirectPred ) / np.pi * 2.0
    shadingIndirectPred = indirectLightNet(
            albedoDS.detach(),
            normalDS.detach(),
            depthDS.detach(),
            shadingDirectPredInput.detach(),
            onMaskSmallBatch )

    shadingPred = shadingIndirectPred + shadingDirectPred
    shadingPred = shadingPred * envMaskSmallBatch + (1 - envMaskSmallBatch) * imSmallBatch

    renderedPred = torch.clamp(shadingPred * albedoDS, 0, 1 )
    renderedPred = renderedPred * (1 - onMaskSmallBatch ) + onMaskSmallBatch * imSmallBatch
    timestop.record()
    torch.cuda.synchronize()
    print('Indirect time: %.3f ms' % timestart.elapsed_time(timestop ) )


    # Save light source output
    for n in range(0, visLampNum ):
        lampName = lampMaskNames[n].replace(inputDir, outputDir )
        lampName = lampName.replace('Mask', 'Src').replace('.png', '.dat').replace('lamp', 'visLamp')

        visLampCenter_np = visLampCenterPreds[n].detach().cpu().numpy().reshape(1, 3)
        visLampSrc_np = visLampSrcPreds[n].detach().cpu().numpy().reshape(1, 3)

        with open(lampName, 'wb') as fOut:
            lampInfo = {
                'center': visLampCenter_np,
                'src': visLampSrc_np
            }
            pickle.dump(lampInfo, fOut )

        visLampShadingPred = visLampShadingPreds[0:1, n, :].detach().cpu().numpy()
        visLampShadingPredIm = visLampShadingPred.squeeze().transpose(1, 2, 0 )
        visLampShadingNoPred = visLampShadingNoPreds[0:1, n, :].detach().cpu().numpy()
        visLampShadingNoPredIm = visLampShadingNoPred.squeeze().transpose(1, 2, 0 )
        visLampShadowPred = visLampShadowPreds[0:1, n, :, :].detach().cpu().numpy()
        visLampShadowPredIm = (np.clip(visLampShadowPred, 0, 1).squeeze() * 255  ).astype(np.uint8 )
        visLampShadowInit = visLampShadowInits[0:1, n, :, :].detach().cpu().numpy()
        visLampShadowInitIm = (np.clip(visLampShadowInit, 0, 1).squeeze() * 255  ).astype(np.uint8 )

        visLampShadingName = osp.join(outputDir, 'visLampShading_%d.npy' % n )
        visLampShadingHdrName = osp.join(outputDir, 'visLampShading_%d.hdr' % n )
        np.save(visLampShadingName, visLampShadingPred )
        cv2.imwrite(visLampShadingHdrName, visLampShadingPredIm[:, :, ::-1] )

        visLampShadingNoName = osp.join(outputDir, 'visLampShadingNo_%d.npy' % n )
        visLampShadingNoHdrName = osp.join(outputDir, 'visLampShadingNo_%d.hdr' % n )
        np.save(visLampShadingNoName, visLampShadingNoPred )
        cv2.imwrite(visLampShadingNoHdrName, visLampShadingNoPredIm[:, :, ::-1] )

        visLampShadowName = osp.join(outputDir, 'visLampShadow_%d.npy' % n )
        visLampShadowPngName = osp.join(outputDir, 'visLampShadow_%d.png' % n )
        np.save(visLampShadowName, visLampShadowPred )
        cv2.imwrite(visLampShadowPngName, visLampShadowPredIm )

        visLampShadowInitName = osp.join(outputDir, 'visLampShadowInit_%d.npy' % n )
        visLampShadowInitPngName = osp.join(outputDir, 'visLampShadowInit_%d.png' % n )
        np.save(visLampShadowInitName, visLampShadowInit )
        cv2.imwrite(visLampShadowInitPngName, visLampShadowInitIm )


    for n in range(0, visWinNum ):
        winName = winMaskNames[n].replace(inputDir, outputDir )
        winName = winName.replace('Mask', 'Src').replace('.png', '.dat').replace('win', 'visWin')

        visWinCenter_np = visWinCenterPreds[n].detach().cpu().numpy().reshape(1, 3)
        visWinNormal_np = visWinNormalPreds[n].detach().cpu().numpy().reshape(1, 3)
        visWinXAxis_np = visWinXAxisPreds[n].detach().cpu().numpy().reshape(1, 3)
        visWinYAxis_np = visWinYAxisPreds[n].detach().cpu().numpy().reshape(1, 3)

        visWinSrc_np = visWinSrcPreds[n].detach().cpu().numpy().reshape(1, 7 )
        visWinSrcSky_np = visWinSrcSkyPreds[n].detach().cpu().numpy().reshape(1, 7 )
        visWinSrcGrd_np = visWinSrcGrdPreds[n].detach().cpu().numpy().reshape(1, 7 )

        with open(winName, 'wb') as fOut:
            winInfo = {
                'xAxis': visWinXAxis_np,
                'yAxis': visWinYAxis_np,
                'normal': visWinNormal_np,
                'center': visWinCenter_np,
                'src': visWinSrc_np,
                'srcSky': visWinSrcSky_np,
                'srcGrd': visWinSrcGrd_np
            }
            pickle.dump(winInfo, fOut )

        visWinShadingPred = visWinShadingPreds[0:1, n, :].detach().cpu().numpy()
        visWinShadingPredIm = visWinShadingPred.squeeze().transpose(1, 2, 0 )
        visWinShadingNoPred = visWinShadingNoPreds[0:1, n, :].detach().cpu().numpy()
        visWinShadingNoPredIm = visWinShadingNoPred.squeeze().transpose(1, 2, 0 )
        visWinShadowPred = visWinShadowPreds[0:1, n, :, :].detach().cpu().numpy()
        visWinShadowPredIm = (np.clip(visWinShadowPred, 0, 1).squeeze() * 255  ).astype(np.uint8 )
        visWinShadowInit = visWinShadowInits[0:1, n, :, :].detach().cpu().numpy()
        visWinShadowInitIm = (np.clip(visWinShadowInit, 0, 1).squeeze() * 255  ).astype(np.uint8 )

        visWinShadingName = osp.join(outputDir, 'visWinShading_%d.npy' % n )
        visWinShadingHdrName = osp.join(outputDir, 'visWinShading_%d.hdr' % n )
        np.save(visWinShadingName, visWinShadingPred )
        cv2.imwrite(visWinShadingHdrName, visWinShadingPredIm[:, :, ::-1] )

        visWinShadingNoName = osp.join(outputDir, 'visWinShadingNo_%d.npy' % n )
        visWinShadingNoHdrName = osp.join(outputDir, 'visWinShadingNo_%d.hdr' % n )
        np.save(visWinShadingNoName, visWinShadingNoPred )
        cv2.imwrite(visWinShadingNoHdrName, visWinShadingNoPredIm[:, :, ::-1] )

        visWinShadowName = osp.join(outputDir, 'visWinShadow_%d.npy' % n )
        visWinShadowPngName = osp.join(outputDir, 'visWinShadow_%d.png' % n )
        np.save(visWinShadowName, visWinShadowPred )
        cv2.imwrite(visWinShadowPngName, visWinShadowPredIm )

        visWinShadowInitName = osp.join(outputDir, 'visWinShadowInit_%d.npy' % n )
        visWinShadowInitPngName = osp.join(outputDir, 'visWinShadowInit_%d.png' % n )
        np.save(visWinShadowInitName, visWinShadowInit )
        cv2.imwrite(visWinShadowInitPngName, visWinShadowInitIm )

    invLampAxes_np = invLampAxesPred.detach().cpu().numpy().reshape(1, 3, 3)
    invLampCenter_np = invLampCenterPred.detach().cpu().numpy().reshape(1, 3)
    invLampSrc_np = invLampSrcPred.detach().cpu().numpy().reshape(1, 3)

    lampName = osp.join(outputDir, 'invLampSrc.dat' )
    with open(lampName, 'wb') as fOut:
        lampInfo = {
            'axes': invLampAxes_np,
            'center': invLampCenter_np,
            'src': invLampSrc_np
        }
        pickle.dump(lampInfo, fOut )

    invLampShadingPred = invLampShadingPred.detach().cpu().numpy()
    invLampShadingPredIm = invLampShadingPred.squeeze().transpose(1, 2, 0 )
    invLampShadingNoPred = invLampShadingNoPred.detach().cpu().numpy()
    invLampShadingNoPredIm = invLampShadingNoPred.squeeze().transpose(1, 2, 0 )
    invLampShadowPred = invLampShadowPred.detach().cpu().numpy()
    invLampShadowPredIm = (np.clip(invLampShadowPred, 0, 1).squeeze() * 255  ).astype(np.uint8 )
    invLampShadowInit = invLampShadowInit.detach().cpu().numpy()
    invLampShadowInitIm = (np.clip(invLampShadowInit, 0, 1).squeeze() * 255  ).astype(np.uint8 )

    invLampShadingName = osp.join(outputDir, 'invLampShading.npy' )
    invLampShadingHdrName = osp.join(outputDir, 'invLampShading.hdr' )
    np.save(invLampShadingName, invLampShadingPred )
    cv2.imwrite(invLampShadingHdrName, invLampShadingPredIm[:, :, ::-1] )

    invLampShadingNoName = osp.join(outputDir, 'invLampShadingNo.npy' )
    invLampShadingNoHdrName = osp.join(outputDir, 'invLampShadingNo.hdr' )
    np.save(invLampShadingNoName, invLampShadingNoPred )
    cv2.imwrite(invLampShadingNoHdrName, invLampShadingNoPredIm[:, :, ::-1] )

    invLampShadowName = osp.join(outputDir, 'invLampShadow.npy' )
    invLampShadowPngName = osp.join(outputDir, 'invLampShadow.png' )
    np.save(invLampShadowName, invLampShadowPred )
    cv2.imwrite(invLampShadowPngName, invLampShadowPredIm )

    invLampShadowInitName = osp.join(outputDir, 'invLampShadowInit.npy' )
    invLampShadowInitPngName = osp.join(outputDir, 'invLampShadowInit.png' )
    np.save(invLampShadowInitName, invLampShadowInit )
    cv2.imwrite(invLampShadowInitPngName, invLampShadowInitIm )

    invWinCenter_np = invWinCenterPred.detach().cpu().numpy().reshape(1, 3 )
    invWinNormal_np = invWinNormalPred.detach().cpu().numpy().reshape(1, 3 )
    invWinXAxis_np = invWinXAxisPred.detach().cpu().numpy().reshape(1, 3 )
    invWinYAxis_np = invWinYAxisPred.detach().cpu().numpy().reshape(1, 3 )

    invWinSrc_np = invWinSrcPred.detach().cpu().numpy().reshape(1, 7 )
    invWinSrcSky_np = invWinSrcSkyPred.detach().cpu().numpy().reshape(1, 7 )
    invWinSrcGrd_np = invWinSrcGrdPred.detach().cpu().numpy().reshape(1, 7 )

    invWinShadingPred = invWinShadingPred.detach().cpu().numpy()
    invWinShadingPredIm = invWinShadingPred.squeeze().transpose(1, 2, 0 )
    invWinShadingNoPred = invWinShadingNoPred.detach().cpu().numpy()
    invWinShadingNoPredIm = invWinShadingNoPred.squeeze().transpose(1, 2, 0 )
    invWinShadowPred = invWinShadowPred.detach().cpu().numpy()
    invWinShadowPredIm = (np.clip(invWinShadowPred, 0, 1).squeeze() * 255  ).astype(np.uint8 )
    invWinShadowInit = invWinShadowInit.detach().cpu().numpy()
    invWinShadowInitIm = (np.clip(invWinShadowInit, 0, 1).squeeze() * 255  ).astype(np.uint8 )

    invWinShadingName = osp.join(outputDir, 'invWinShading.npy' )
    invWinShadingHdrName = osp.join(outputDir, 'invWinShading.hdr' )
    np.save(invWinShadingName, invWinShadingPred )
    cv2.imwrite(invWinShadingHdrName, invWinShadingPredIm[:, :, ::-1] )

    invWinShadingNoName = osp.join(outputDir, 'invWinShadingNo.npy' )
    invWinShadingNoHdrName = osp.join(outputDir, 'invWinShadingNo.hdr' )
    np.save(invWinShadingNoName, invWinShadingNoPred )
    cv2.imwrite(invWinShadingNoHdrName, invWinShadingNoPredIm[:, :, ::-1] )

    invWinShadowName = osp.join(outputDir, 'invWinShadow.npy' )
    invWinShadowPngName = osp.join(outputDir, 'invWinShadow.png' )
    np.save(invWinShadowName, invWinShadowPred )
    cv2.imwrite(invWinShadowPngName, invWinShadowPredIm )

    invWinShadowInitName = osp.join(outputDir, 'invWinShadowInit.npy' )
    invWinShadowInitPngName = osp.join(outputDir, 'invWinShadowInit.png' )
    np.save(invWinShadowInitName, invWinShadowInit )
    cv2.imwrite(invWinShadowInitPngName, invWinShadowInitIm )

    winName = osp.join(outputDir, 'invWinSrc.dat' )
    with open(winName, 'wb') as fOut:
        winInfo = {
            'xAxis': invWinXAxis_np,
            'yAxis': invWinYAxis_np,
            'normal': invWinNormal_np,
            'center': invWinCenter_np,
            'src': invWinSrc_np,
            'srcSky': invWinSrcSky_np,
            'srcGrd': invWinSrcGrd_np
        }
        pickle.dump(winInfo, fOut )

    # Save the BRDF predictions
    albedoName = osp.join(outputDir, 'albedo.npy' )
    albedoImName = osp.join(outputDir, 'albedo.png' )
    albedoPred = albedoPred.detach().cpu().numpy()
    albedoPredIm = albedoPred.squeeze().transpose(1, 2, 0)
    albedoPredIm = ( (albedoPredIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(albedoName, albedoPred )
    cv2.imwrite(albedoImName, albedoPredIm[:, :, ::-1] )

    albedoDSName = osp.join(outputDir, 'albedoDS.npy')
    albedoDSImName = osp.join(outputDir, 'albedoDS.png')
    albedoDS = albedoDS.detach().cpu().numpy()
    albedoDSIm = albedoDS.squeeze().transpose(1, 2, 0)
    albedoDSIm = ( (albedoDSIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(albedoDSName, albedoDS )
    cv2.imwrite(albedoDSImName, albedoDSIm[:, :, ::-1] )

    roughName = osp.join(outputDir, 'rough.npy')
    roughImName = osp.join(outputDir, 'rough.png')
    roughPred = roughPred.detach().cpu().numpy()
    roughPredIm = roughPred.squeeze()
    roughPredIm = ( (roughPredIm+1) * 0.5 * 255).astype(np.uint8 )
    np.save(roughName, roughPred )
    cv2.imwrite(roughImName, roughPredIm )

    roughDSName = osp.join(outputDir, 'roughDS.npy')
    roughDSImName = osp.join(outputDir, 'roughDS.png')
    roughDS = roughDS.detach().cpu().numpy()
    roughDSIm = roughDS.squeeze()
    roughDSIm = ( ( roughDSIm + 1) * 0.5  * 255).astype(np.uint8 )
    np.save(roughDSName, roughDS )
    cv2.imwrite(roughDSImName, roughDSIm )

    normalName = osp.join(outputDir, 'normal.npy')
    normalImName = osp.join(outputDir, 'normal.png')
    normalPred = normalPred.detach().cpu().numpy()
    normalPredIm = normalPred.squeeze().transpose(1, 2, 0 )
    normalPredIm = ( 0.5*(normalPredIm + 1)*255 ).astype(np.uint8 )
    np.save(normalName, normalPred )
    cv2.imwrite(normalImName, normalPredIm[:, :, ::-1] )

    normalDSName = osp.join(outputDir, 'normalDS.npy')
    normalDSImName = osp.join(outputDir, 'normalDS.png')
    normalDS = normalDS.detach().cpu().numpy()
    normalDSIm = normalDS.squeeze().transpose(1, 2, 0 )
    normalDSIm = ( 0.5*(normalDSIm + 1)*255 ).astype(np.uint8 )
    np.save(normalDSName, normalDS )
    cv2.imwrite(normalDSImName, normalDSIm[:, :, ::-1] )

    depthName = osp.join(outputDir, 'depth.npy')
    depthImName = osp.join(outputDir, 'depth.png')
    depthPred = depthPred.detach().cpu().numpy()
    depthPredIm = depthPred.squeeze()
    depthPredIm = ( 1 / (depthPredIm + 1)  * 255).astype(np.uint8 )
    np.save(depthName, depthPred )
    cv2.imwrite(depthImName, depthPredIm )

    depthDSName = osp.join(outputDir, 'depthDS.npy')
    depthDSImName = osp.join(outputDir, 'depthDS.png')
    depthDS = depthDS.detach().cpu().numpy()
    depthDSIm = depthDS.squeeze()
    depthDSIm = ( 1 / (depthDSIm + 1)  * 255).astype(np.uint8 )
    np.save(depthDSName, depthDS )
    cv2.imwrite(depthDSImName, depthDSIm )

    # Save the images
    imName = osp.join(outputDir, 'im.npy' )
    imPngName = osp.join(outputDir, 'im.png' )
    imBatch = imBatch.detach().cpu().numpy()
    imBatchIm = imBatch.squeeze().transpose(1, 2, 0)
    imBatchIm = ( (imBatchIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(imName, imBatch )
    cv2.imwrite(imPngName, imBatchIm[:, :, ::-1] )

    imSmallName = osp.join(outputDir, 'imSmall.npy' )
    imSmallPngName = osp.join(outputDir, 'imSmall.png' )
    imSmallBatch = imSmallBatch.detach().cpu().numpy()
    imSmallBatchIm = imSmallBatch.squeeze().transpose(1, 2, 0)
    imSmallBatchIm = ( (imSmallBatchIm **(1.0/2.2) ) * 255).astype(np.uint8 )
    np.save(imSmallName, imSmallBatch )
    cv2.imwrite(imSmallPngName, imSmallBatchIm[:, :, ::-1] )

    renderedName = osp.join(outputDir, 'rendered.npy')
    renderedHdrName = osp.join(outputDir, 'rendered.hdr')
    renderedPred = renderedPred.detach().cpu().numpy()
    renderedPredIm = renderedPred.squeeze().transpose(1, 2, 0 )
    np.save(renderedName, renderedPred )
    cv2.imwrite(renderedHdrName, renderedPredIm[:, :, ::-1])

    shadingDirectName = osp.join(outputDir, 'shadingDirect.npy')
    shadingDirectHdrName = osp.join(outputDir, 'shadingDirect.hdr')
    shadingDirectPred = shadingDirectPred.detach().cpu().numpy()
    shadingDirectPredIm = shadingDirectPred.squeeze().transpose(1, 2, 0 )
    np.save(shadingDirectName, shadingDirectPred )
    cv2.imwrite(shadingDirectHdrName, shadingDirectPredIm[:, :, ::-1])

    shadingName = osp.join(outputDir, 'shading.npy')
    shadingHdrName = osp.join(outputDir, 'shading.hdr')
    shadingPred = shadingPred.detach().cpu().numpy()
    shadingPredIm = shadingPred.squeeze().transpose(1, 2, 0 )
    np.save(shadingName, shadingPred )
    cv2.imwrite(shadingHdrName, shadingPredIm[:, :, ::-1])

    # Save the light source on mask
    envMaskName = osp.join(outputDir, 'envMask.npy')
    envMaskImName = osp.join(outputDir, 'envMask.png')
    envMaskBatch = envMaskBatch.detach().cpu().numpy()
    envMaskBatchIm = envMaskBatch.squeeze()
    envMaskBatchIm = ( envMaskBatchIm  * 255).astype(np.uint8 )
    np.save(envMaskName, envMaskBatch )
    cv2.imwrite(envMaskImName, envMaskBatchIm )

    envMaskSmallName = osp.join(outputDir, 'envMaskSmall.npy')
    envMaskSmallImName = osp.join(outputDir, 'envMaskSmall.png')
    envMaskSmallBatch = envMaskSmallBatch.detach().cpu().numpy()
    envMaskSmallBatchIm = envMaskSmallBatch.squeeze()
    envMaskSmallBatchIm = ( envMaskSmallBatchIm  * 255).astype(np.uint8 )
    np.save(envMaskSmallName, envMaskSmallBatch )
    cv2.imwrite(envMaskSmallImName, envMaskSmallBatchIm )

    # Save the environment mask
    onMaskName = osp.join(outputDir, 'onMask.npy')
    onMaskImName = osp.join(outputDir, 'onMask.png')
    onMaskBatch = onMaskBatch.detach().cpu().numpy()
    onMaskBatchIm = onMaskBatch.squeeze()
    onMaskBatchIm = ( onMaskBatchIm  * 255).astype(np.uint8 )
    np.save(onMaskName, onMaskBatch )
    cv2.imwrite(onMaskImName, onMaskBatchIm )

    onMaskSmallName = osp.join(outputDir, 'onMaskSmall.npy')
    onMaskSmallImName = osp.join(outputDir, 'onMaskSmall.png')
    onMaskSmallBatch = onMaskSmallBatch.detach().cpu().numpy()
    onMaskSmallBatchIm = onMaskSmallBatch.squeeze()
    onMaskSmallBatchIm = ( onMaskSmallBatchIm  * 255).astype(np.uint8 )
    np.save(onMaskSmallName, onMaskSmallBatch )
    cv2.imwrite(onMaskSmallImName, onMaskSmallBatchIm )
