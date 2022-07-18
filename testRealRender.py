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

        visWinShadowPred = visWinShadowPreds[:, n]
        if visWinShadowPred.size(2 ) != visWinShadingNoPred.size(2 ):
            visWinShadowPred = F.interpolate(visWinShadowPred, visWinShadingNoPred.size()[2:4], mode='bilinear')

        visWinShadingPreds.append(visWinShadingNoPred * visWinShadowPred )

    return visWinShadingPreds, visWinShadingNoPreds


def renderVisLampArr(
    visLampCenterPreds, visLampSrcPreds, onLampMasksSmallBatch,
    depthDS, normalDS,
    visLampShadowPreds, visLampMeshNames ):

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
                isTest = False,
                visLampMeshNames = [visLampMeshNames[n]] if visLampMeshNames is not None else None )

        visLampShadingNoPreds.append(visLampShadingNoPred )

        visLampShadingNoPredSelfOcclu, visLampPointsPred \
                = renderVisLamp.forward(
                visLampCenterPreds[n],
                visLampSrcPreds[n],
                depthDS.detach(),
                onLampMasksSmallBatch[:, n:n+1, :],
                normalDS.detach(),
                isTest = True,
                visLampMeshNames = [visLampMeshNames[n]] if visLampMeshNames is not None else None )

        visLampShadowPred = visLampShadowPreds[:, n]
        if visLampShadowPred.size(2 ) != visLampShadingNoPred.size(2 ):
            visLampShadowPred = F.interpolate(visLampShadowPred, visLampShadingNoPred.size()[2:4], mode='bilinear')

        visLampShadingPreds.append(visLampShadingNoPredSelfOcclu * visLampShadowPred )

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

    if invWinShadowPred.size(2 ) != invWinShadingNoPred.size(2 ):
        invWinShadowPred = F.interpolate(invWinShadowPred, invWinShadingNoPred.size()[2:4], mode='bilinear')
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

    if invLampShadowPred.size(2 ) != invLampShadingNoPred.size(2 ):
        invLampShadowPred = F.interpolate(invLampShadowPred, invLampShadingNoPred.size()[2:4], mode='bilinear')
    invLampShadingPred = invLampShadingNoPredSelfOcclu * invLampShadowPred

    return invLampShadingPred, invLampShadingNoPred


parser = argparse.ArgumentParser()
# The directory of trained models
parser.add_argument('--experimentDirecIndirec', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentShadow', default=None, help='the path to store samples and models')
parser.add_argument('--experimentShgEnv', default=None, help='the path to store samples and models' )
parser.add_argument('--testList', default=None, help='the path to store samples and models' )

# The basic training setting
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=8, help='the size of envmaps in y direction')
parser.add_argument('--envWidth', type=int, default=16, help='the size of envmaps in x direction')
parser.add_argument('--objName', default=None, help='the obj file inserted' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')

# Training epochs and iterations
parser.add_argument('--iterIdVisLamp', type=int, default=119540, help='the iteration used for testing' )
parser.add_argument('--iterIdInvLamp', type=int, default=150000, help='the iteration used for testing' )
parser.add_argument('--iterIdVisWin', type=int, default=120000, help='the iteration used for testing' )
parser.add_argument('--iterIdInvWin', type=int, default=200000, help='the iteration used for testing' )
parser.add_argument('--iterIdShadow', type=int, default=70000, help='the iteration used for testing')
parser.add_argument('--iterIdDirecIndirec', type=int, default=180000, help='the iteration used for testing' )
parser.add_argument('--iterIdShgEnv', type=int, default=240000, help='the iteration used for testing' )

parser.add_argument('--isOptimize', action='store_true', help='use optimization for light sources or not' )
parser.add_argument('--isPerpixelLighting', action='store_true', help='whether to output per-pixel lighting or not')
parser.add_argument('--isVisLampMesh', action='store_true', help='whether to directly load light source mesh')

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
parser.add_argument('--re', type=int, default=100, help='ending point' )

opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )

if opt.experimentDirecIndirec is None:
    opt.experimentDirecIndirec = 'check_directIndirect'
opt.experimentDirecIndirec = osp.join(curDir, opt.experimentDirecIndirec )

if opt.experimentShadow is None:
    opt.experimentShadow = 'check_shadowDepth'
opt.experimentShadow = osp.join(curDir, opt.experimentShadow )

if opt.experimentShgEnv is None:
    opt.experimentShgEnv = 'check_shadingToLight'
opt.experimentShgEnv = osp.join(curDir, opt.experimentShgEnv )

indirectLightNet = modelLight.indirectLightNet()
indirectLightDict = torch.load(
    '{0}/indirectLightNet_iter{1}.pth'.format(opt.experimentDirecIndirec, opt.iterIdDirecIndirec ) )
indirectLightNet.load_state_dict(indirectLightDict['model'] )
for param in indirectLightNet.parameters():
    param.requires_grad = False
indirectLightNet = indirectLightNet.cuda()

lightEncoder = modelLight.encoderLight()
weightDecoder = modelLight.decoderLight(mode = 2 )
axisDecoder = modelLight.decoderLight(mode = 0 )
lambDecoder = modelLight.decoderLight(mode = 1 )
lightEncoder.load_state_dict(
    torch.load('{0}/lightEncoder_iter{1}.pth'.format(opt.experimentShgEnv, opt.iterIdShgEnv ) )['model'] )
weightDecoder.load_state_dict(
    torch.load('{0}/weightDecoder_iter{1}.pth'.format(opt.experimentShgEnv, opt.iterIdShgEnv ) )['model'] )
axisDecoder.load_state_dict(
    torch.load('{0}/axisDecoder_iter{1}.pth'.format(opt.experimentShgEnv, opt.iterIdShgEnv ) )['model'] )
lambDecoder.load_state_dict(
    torch.load('{0}/lambDecoder_iter{1}.pth'.format(opt.experimentShgEnv, opt.iterIdShgEnv ) )['model'] )
for param in lightEncoder.parameters():
    param.requires_grad = False
for param in weightDecoder.parameters():
    param.requires_grad = False
for param in axisDecoder.parameters():
    param.requires_grad = False
for param in lambDecoder.parameters():
    param.requires_grad = False
lightEncoder = lightEncoder.cuda()
weightDecoder = weightDecoder.cuda()
axisDecoder = axisDecoder.cuda()
lambDecoder = lambDecoder.cuda()

renderWindow = renderWindow.renderDirecLighting(sampleNum = 400 )
renderVisLamp = renderVisLamp.renderDirecLighting(max_plate = 512 )
renderInvLamp = renderInvLamp.renderDirecLighting(sampleNum = 10 )
renderShadow = renderShadowDepth.renderShadow(
    modelRoot = opt.experimentShadow, iterId = opt.iterIdShadow,
    winSampleNum = 1024, lampSampleNum = 1024
)

output2env = models.output2env(envWidth = opt.envWidth, envHeight = opt.envHeight )
####################################
with open(opt.testList, 'r') as fIn:
    dirList = fIn.readlines()
dirList = [x.strip() for x in dirList ]

timestart = torch.cuda.Event(enable_timing = True )
timestop = torch.cuda.Event(enable_timing = True )

for dataId in range(max(opt.rs, 0), min(opt.re, len(dirList ) ) ):
    dataDir = dirList[dataId ]

    renderShadow.setOutputRoot(dataDir )
    print(dataDir )
    if opt.objName is not None:
        objName = osp.join(dataDir, opt.objName )
    else:
        objName = None

    roomName = osp.join(dataDir, 'room.obj')

    rawDir = osp.join(dataDir, 'input')
    inputDir = osp.join(dataDir, 'BRDFLight')
    inputDir += '_size%.3f_int%.3f_dir%.3f_lam%.3f_ren%.3f' \
        % (opt.sizeWeight, opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight,
           opt.renderWeight )
    inputDir += '_visWin%d_visLamp%d_invWin%d_invLamp%d' \
        % (opt.iterIdVisWin, opt.iterIdVisLamp, opt.iterIdInvWin, opt.iterIdInvLamp )
    if opt.isOptimize:
        inputDir += '_optimize'
    outputDir = inputDir.replace('BRDFLight', 'Rendering')

    if not osp.isdir(outputDir ):
        os.system('mkdir %s' % outputDir )

    imName = osp.join(inputDir, 'im.npy')
    imBatch = torch.from_numpy(np.load(imName ) ).cuda()
    imSmallName = osp.join(inputDir, 'imSmall.npy')
    imSmallBatch = torch.from_numpy(np.load(imSmallName ) ).cuda()

    height, width = imBatch.size()[2:]
    sHeight, sWidth = imSmallBatch.size()[2:]


    lampMaskNames = glob.glob(osp.join(rawDir, 'lampMask_*.png') )
    if len(lampMaskNames ) > 1:
        lampMaskNames = sorted(lampMaskNames )
    visLampNum = len(lampMaskNames )
    if opt.isVisLampMesh:
        visLampMeshNames = []
        for n in range(0, visLampNum ):
            visLampMeshNames.append(osp.join(rawDir, 'visLamp_%d.obj' % n ) )
    else:
        visLampMeshNames = None

    winMaskNames = glob.glob(osp.join(rawDir, 'winMask_*.png') )
    if len(winMaskNames ) > 1:
        winMaskNames = sorted(winMaskNames )
    visWinNum = len(winMaskNames )

    renderShadow.setWinNum(visWinNum )
    renderShadow.setLampNum(visLampNum )

    lampMasks, lampMaskSmalls = [], []
    for lampMaskName in lampMaskNames:
        lampMask = cv2.imread(lampMaskName )
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
        winMask = cv2.resize(winMask, (width, height), interpolation = cv2.INTER_AREA )
        winMaskSmall = cv2.resize(winMask, (sWidth, sHeight), interpolation = cv2.INTER_AREA )
        if len(winMask.shape ) == 3:
            winMask = winMask[:, :, 0]
            winMaskSmall = winMaskSmall[:, :, 0]
        winMasks.append(winMask )
        winMaskSmalls.append(winMaskSmall )

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


    albedoName = osp.join(inputDir, 'albedo.npy')
    albedoBatch = torch.from_numpy(np.load(albedoName ) ).cuda()
    albedoDSName = osp.join(inputDir, 'albedoDS.npy')
    albedoDS = torch.from_numpy(np.load(albedoDSName ) ).cuda()

    normalName = osp.join(inputDir, 'normal.npy')
    normalBatch = torch.from_numpy(np.load(normalName ) ).cuda()
    normalDSName = osp.join(inputDir, 'normalDS.npy')
    normalDS = torch.from_numpy(np.load(normalDSName ) ).cuda()

    roughName = osp.join(inputDir, 'rough.npy')
    roughBatch = torch.from_numpy(np.load(roughName ) ).cuda()
    roughDSName = osp.join(inputDir, 'roughDS.npy')
    roughDS = torch.from_numpy(np.load(roughDSName ) ).cuda()

    if opt.objName is None:
        depthName = osp.join(inputDir, 'depth.npy')
        depthBatch = torch.from_numpy(np.load(depthName ) ).cuda()
        depthDSName = osp.join(inputDir, 'depthDS.npy')
        depthDS = torch.from_numpy(np.load(depthDSName ) ).cuda()
    else:
        depthName = osp.join(inputDir, 'depth.npy')
        depthBatchOrigin = torch.from_numpy(np.load(depthName ) ).cuda()
        depthDSName = osp.join(inputDir, 'depthDS.npy')
        depthDSOrigin = torch.from_numpy(np.load(depthDSName ) ).cuda()

        depthName = osp.join(inputDir, 'depthBlend.npy')
        depthBatch = torch.from_numpy(np.load(depthName ) ).cuda()
        depthDSName = osp.join(inputDir, 'depthDSBlend.npy')
        depthDS = torch.from_numpy(np.load(depthDSName ) ).cuda()

    envMaskName = osp.join(inputDir, 'envMask.npy')
    envMaskBatch = torch.from_numpy(np.load(envMaskName ) ).cuda()
    envMaskSmallName = osp.join(inputDir, 'envMaskSmall.npy')
    envMaskSmallBatch = torch.from_numpy(np.load(envMaskSmallName ) ).cuda()

    onMaskName = osp.join(inputDir, 'onMask.npy')
    onMaskBatch = torch.from_numpy(np.load(onMaskName ) ).cuda()
    onMaskSmallName = osp.join(inputDir, 'onMaskSmall.npy')
    onMaskSmallBatch = torch.from_numpy(np.load(onMaskSmallName ) ).cuda()

    # Load visible lamps
    visLampCenterPreds, visLampSrcPreds = [], []
    for n in range(0, visLampNum ):
        lampName = lampMaskNames[n].replace(rawDir, inputDir )
        lampName = lampName.replace('Mask', 'Src').replace('.png', '.dat').replace('lamp', 'visLamp')
        with open(lampName, 'rb') as fIn:
            lampInfo = pickle.load(fIn )

        lampCenter = torch.from_numpy(lampInfo['center'] ).cuda()
        lampSrc = torch.from_numpy(lampInfo['src'] ).cuda()

        visLampCenterPreds.append(lampCenter )
        visLampSrcPreds.append(lampSrc )

    # Load visible windows
    visWinCenterPreds = []
    visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds = [], [], []
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds = [], [], []
    for n in range(0, visWinNum ):
        winName = winMaskNames[n].replace(rawDir, inputDir )
        winName = winName.replace('Mask', 'Src').replace('.png', '.dat').replace('win', 'visWin')
        with open(winName, 'rb') as fIn:
            winInfo = pickle.load(fIn )

        winXAxis = torch.from_numpy(winInfo['xAxis'] ).cuda()
        winYAxis = torch.from_numpy(winInfo['yAxis'] ).cuda()
        winNormal = torch.from_numpy(winInfo['normal'] ).cuda()
        winCenter = torch.from_numpy(winInfo['center'] ).cuda()

        winSrc = torch.from_numpy(winInfo['src'] ).cuda()
        winSrcSky = torch.from_numpy(winInfo['srcSky'] ).cuda()
        winSrcGrd = torch.from_numpy(winInfo['srcGrd'] ).cuda()

        visWinCenterPreds.append(winCenter )
        visWinNormalPreds.append(winNormal )
        visWinXAxisPreds.append(winXAxis )
        visWinYAxisPreds.append(winYAxis )

        visWinSrcPreds.append(winSrc )
        visWinSrcSkyPreds.append(winSrcSky )
        visWinSrcGrdPreds.append(winSrcGrd )

    # Load invisble lamps
    invLampName = osp.join(inputDir, 'invLampSrc.dat' )
    with open(invLampName, 'rb') as fIn:
        lampInfo = pickle.load(fIn )

    invLampCenterPred = torch.from_numpy(lampInfo['center'] ).cuda()
    invLampAxesPred = torch.from_numpy(lampInfo['axes' ] ).cuda()
    invLampSrcPred = torch.from_numpy(lampInfo['src'] ).cuda()

    invWinName = osp.join(inputDir, 'invWinSrc.dat' )
    with open(invWinName, 'rb') as fIn:
        winInfo = pickle.load(fIn )

    invWinXAxisPred = torch.from_numpy(winInfo['xAxis'] ).cuda()
    invWinYAxisPred = torch.from_numpy(winInfo['yAxis'] ).cuda()
    invWinNormalPred = torch.from_numpy(winInfo['normal'] ).cuda()
    invWinCenterPred = torch.from_numpy(winInfo['center'] ).cuda()

    invWinSrcPred = torch.from_numpy(winInfo['src'] ).cuda()
    invWinSrcSkyPred = torch.from_numpy(winInfo['srcSky'] ).cuda()
    invWinSrcGrdPred = torch.from_numpy(winInfo['srcGrd'] ).cuda()

    if opt.objName is None:
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
                objName= objName, roomName = roomName, visLampMeshNames = visLampMeshNames )
    else:
        visWinShadowInits, visWinShadowPreds, \
            visLampShadowInits, visLampShadowPreds, \
            invWinShadowInit, invWinShadowPred, \
            invLampShadowInit, invLampShadowPred \
            = renderShadow.forward(
                depthDSOrigin, normalDS,
                visWinCenterPreds, visWinXAxisPreds, visWinYAxisPreds, winMaskSmallBatch,
                visLampCenterPreds, lampMaskSmallBatch,
                invWinCenterPred, invWinXAxisPred, invWinYAxisPred,
                invLampAxesPred, invLampCenterPred,
                objName = objName, roomName = roomName, visLampMeshNames = visLampMeshNames )

    # Compute rendering error for visible window
    visWinShadingPreds, visWinShadingNoPreds, \
        = renderVisWindowArr(
            # Visible window parameters
            visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
            visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
            depthDS if not opt.isHighRes else depthBatch,
            normalDS if not opt.isHighRes else normalBatch,
            visWinShadowPreds
        )

    # Compute rendering error for visible lamp
    visLampShadingPreds, visLampShadingNoPreds, \
        = renderVisLampArr(
            visLampCenterPreds, visLampSrcPreds,
            lampMaskSmallBatch if not opt.isHighRes else lampMaskBatch,
            depthDS if not opt.isHighRes else depthBatch,
            normalDS if not opt.isHighRes else normalBatch,
            visLampShadowPreds, visLampMeshNames
        )

    # Compute rendering error for invisible window
    invWinShadingPred, invWinShadingNoPred, \
        = renderInvWindowArr(
            invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
            depthDS if not opt.isHighRes else depthBatch,
            normalDS if not opt.isHighRes else normalBatch,
            invWinShadowPred
        )

    # Compute rendering error for invisible lamp
    invLampShadingPred, invLampShadingNoPred, \
        = renderInvLampArr(
            invLampAxesPred, invLampCenterPred,
            invLampSrcPred,
            depthDS if not opt.isHighRes else depthBatch,
            normalDS if not opt.isHighRes else normalBatch,
            invLampShadowPred
        )

    # Predict the global illumination
    if len(visWinSrcPreds ) > 0:
        visWinShadingNoPreds = torch.cat(visWinShadingNoPreds, dim=0 ).reshape(
                1, visWinNum, 3, height if opt.isHighRes else sHeight, width if opt.isHighRes else sWidth )
        visWinShadingPreds = torch.cat(visWinShadingPreds, dim=0 ).reshape(
                1, visWinNum, 3, height if opt.isHighRes else sHeight, width if opt.isHighRes else sWidth )

    if len(visLampSrcPreds ) > 0:
        visLampShadingNoPreds = torch.cat(visLampShadingNoPreds, dim=0 ).reshape(
                1, visLampNum, 3, height if opt.isHighRes else sHeight, width if opt.isHighRes else sWidth )
        visLampShadingPreds = torch.cat(visLampShadingPreds, dim=0 ).reshape(
                1, visLampNum, 3, height if opt.isHighRes else sHeight, width if opt.isHighRes else sWidth )

    shadingDirectPred = invWinShadingPred + invLampShadingPred
    if len(visWinSrcPreds ) > 0:
        shadingDirectPred += torch.sum(visWinShadingPreds, dim=1 )

    if len(visLampSrcPreds ) > 0:
        shadingDirectPred += torch.sum(visLampShadingPreds, dim=1 )

    if not opt.isHighRes:
        shadingDirectPred = shadingDirectPred * envMaskSmallBatch + (1 - envMaskSmallBatch) * imSmallBatch
        shadingDirectPred = shadingDirectPred * (1 - onMaskSmallBatch ) + onMaskSmallBatch * imSmallBatch
    else:
        shadingDirectPred = shadingDirectPred * envMaskBatch + (1 - envMaskBatch) * imBatch
        shadingDirectPred = shadingDirectPred * (1 - onMaskBatch) + onMaskBatch * imBatch

    shadingDirectPredInput = torch.atan(shadingDirectPred ) / np.pi * 2.0
    if opt.isHighRes:
        shadingDirectPredInput = F.adaptive_avg_pool2d(shadingDirectPredInput, (sHeight, sWidth ) )


    shadingIndirectPred = indirectLightNet(
            albedoDS.detach(),
            normalDS.detach(),
            depthDS.detach(),
            shadingDirectPredInput.detach(),
            onMaskSmallBatch )

    if opt.isHighRes:
        shadingIndirectPred = F.interpolate(shadingIndirectPred, (height, width), mode = 'bilinear')


    shadingPred = shadingIndirectPred + shadingDirectPred

    if opt.isHighRes:
        shadingPred = shadingPred * envMaskBatch + (1 - envMaskBatch) * imBatch
        shadingPred = shadingPred * (1 - onMaskBatch) + onMaskBatch * imBatch

        renderedPred = torch.clamp(shadingPred * albedoBatch, 0, 1 )
        renderedPred = renderedPred * (1 - onMaskBatch ) + onMaskBatch * imBatch
    else:
        shadingPred = shadingPred * envMaskSmallBatch + (1 - envMaskSmallBatch) * imSmallBatch
        shadingPred = shadingPred * (1 - onMaskSmallBatch) + onMaskSmallBatch * imSmallBatch

        renderedPred = torch.clamp(shadingPred * albedoDS, 0, 1 )
        renderedPred = renderedPred * (1 - onMaskSmallBatch ) + onMaskSmallBatch * imSmallBatch

    # Predict per-pixel lighting
    if opt.isPerpixelLighting:
        timestart.record()
        inputBatch = torch.cat([albedoDS, normalDS, roughDS, depthDS, onMaskSmallBatch ], dim=1 )

        if opt.isHighRes:
            shadingPredInput = F.adaptive_avg_pool2d(shadingPred, (sHeight, sWidth ) )
        else:
            shadingPredInput = shadingPred

        x1, x2, x3, x4, x5, x6 = lightEncoder(inputBatch, shadingPredInput )

        # Prediction
        axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, shadingPredInput )
        lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, shadingPredInput )
        weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, shadingPredInput )
        bn, SGNum, _, envRow, envCol = axisPred.size()

        envPred, axisPred, lambPred, weightPred \
            = output2env.output2env(axisPred, lambPred, weightPred )
        envShadingPred = utils.envToShading(envPred )

        timestop.record()
        torch.cuda.synchronize()
        print('Per-pixel lighting time: %.3f ms' % timestart.elapsed_time(timestop ) )

        envName = osp.join(outputDir, 'env.npy')
        envImName = osp.join(outputDir, 'envPred.hdr')
        envPred = envPred.detach().cpu().numpy()
        np.save(envName, envPred )

        envPredIm = envPred.squeeze().transpose(1, 3, 2, 4, 0 )
        envPredIm = envPredIm.reshape(sHeight * opt.envHeight, sWidth * opt.envWidth, 3 )
        cv2.imwrite(envImName, envPredIm[:, :, ::-1] )

        envShadingName = osp.join(outputDir, 'envShading.npy')
        envShadingHdrName = osp.join(outputDir, 'envShading.hdr')
        envShadingPred = envShadingPred.detach().cpu().numpy()
        envShadingPredIm = envShadingPred.squeeze().transpose(1, 2, 0 )
        np.save(envShadingName, envShadingPred )
        cv2.imwrite(envShadingHdrName, envShadingPredIm[:, :, ::-1] )

    # Save light source output
    for n in range(0, visLampNum ):
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

    # Save geometry
    if opt.isHighRes:
        utils.writeDepthAsPointClouds(
            depthBatch * ( (1 - onMaskBatch) == 1 ).float(),
            renderedPred ** (1 / 2.2 ),
            envMaskBatch,
            osp.join(outputDir, 'room.ply'),
            isNormalize = False
        )
    else:
        utils.writeDepthAsPointClouds(
            depthDS * ( (1 - onMaskSmallBatch) == 1).float(),
            renderedPred ** (1/2.2 ),
            envMaskSmallBatch,
            osp.join(outputDir, 'room.ply'),
            isNormalize = False
        )



    for n in range(0, visWinNum ):
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
    cv2.imwrite(shadingDirectHdrName, shadingDirectPredIm[:, :, ::-1] )

    shadingName = osp.join(outputDir, 'shading.npy')
    shadingHdrName = osp.join(outputDir, 'shading.hdr')
    shadingPred = shadingPred.detach().cpu().numpy()
    shadingPredIm = shadingPred.squeeze().transpose(1, 2, 0 )
    np.save(shadingName, shadingPred )
    cv2.imwrite(shadingHdrName, shadingPredIm[:, :, ::-1])

