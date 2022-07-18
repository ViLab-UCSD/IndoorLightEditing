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


def renderVisWindowArr(
    visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
    depthDS, normalDS, albedoDS, albedoSmallBatch,
    visWinShadowPreds,
    segEnvBatch,
    visWinOns, onVisWinNum, batchSize, maxWinNum ):

    visWinShadingPreds = []
    visWinShadingNoPreds = []

    if onVisWinNum > 0:
        for m in range(0, batchSize ):
            for n in range(0, maxWinNum ):
                if visWinOns[m, n] == 1:
                    visWinShadingNoPred \
                            = renderWindow.forward(
                            visWinCenterPreds[m * maxWinNum + n],
                            visWinNormalPreds[m * maxWinNum + n],
                            visWinYAxisPreds[m * maxWinNum + n],
                            visWinXAxisPreds[m * maxWinNum + n],
                            visWinSrcPreds[m * maxWinNum + n],
                            visWinSrcSkyPreds[m * maxWinNum + n],
                            visWinSrcGrdPreds[m * maxWinNum + n],
                            depthDS.detach()[m:m+1, :],
                            normalDS.detach()[m:m+1, :] )


                    visWinShadingNoPreds.append(visWinShadingNoPred )
                    visWinShadingPreds.append(visWinShadingNoPred * visWinShadowPreds[m:m+1, n] )
                else:
                    visWinShadingNoPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                    visWinShadingPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )


    return visWinShadingPreds, visWinShadingNoPreds


def renderVisLampArr(
    visLampCenterPreds, visLampSrcPreds, onLampMasksSmallBatch,
    depthDS, normalDS, albedoDS, albedoSmallBatch,
    visLampShadowPreds,
    segEnvBatch,
    visLampOns, onVisLampNum, batchSize, maxLampNum ):

    visLampShadingPreds = []
    visLampShadingNoPreds = []
    visLampPointsPreds = []

    visLampShgErr = 0
    visLampRenErr = 0
    visLampPointsErr = 0

    if onVisLampNum > 0:
        for m in range(0, batchSize ):
            for n in range(0, maxLampNum ):
                if visLampOns[m, n] == 1:
                    visLampShadingNoPred, visLampPointsPred \
                            = renderVisLamp.forward(
                            visLampCenterPreds[m * maxLampNum + n],
                            visLampSrcPreds[m * maxLampNum + n],
                            depthDS.detach()[m:m+1, :],
                            onLampMasksSmallBatch[m:m+1, n:n+1, :],
                            normalDS.detach()[m:m+1, :],
                            isTest = False )

                    visLampPointsPreds.append(visLampPointsPred[0] )
                    visLampShadingNoPreds.append(visLampShadingNoPred )

                    visLampShadingNoPredSelfOcclu, visLampPointsPred \
                            = renderVisLamp.forward(
                            visLampCenterPreds[m * maxLampNum + n],
                            visLampSrcPreds[m * maxLampNum + n],
                            depthDS.detach()[m:m+1, :],
                            onLampMasksSmallBatch[m:m+1, n:n+1, :],
                            normalDS.detach()[m:m+1, :],
                            isTest = True )
                    visLampShadingPreds.append(visLampShadingNoPredSelfOcclu * visLampShadowPreds[m:m+1, n, :] )
                else:
                    visLampPointsPreds.append(None )
                    visLampShadingNoPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                    visLampShadingPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )

    return visLampShadingPreds, visLampShadingNoPreds, visLampPointsPreds

def renderInvWindowArr(
    invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
    invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
    depthDS, normalDS, albedoDS, albedoSmallBatch,
    invWinShadowPred,  segEnvBatch ):

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
    depthDS, normalDS, albedoDS, albedoSmallBatch,
    invLampShadowPred, segEnvBatch ):

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


parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images')

parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvLamp', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentVisWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentInvWindow', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentDirecIndirec', default=None, help='the path to store samples and models' )
parser.add_argument('--experimentShadow', default=None, help='the path to store samples and models')
parser.add_argument('--experimentPerpixelEnv', default=None, help='the path to store samples and models')

parser.add_argument('--testRoot', default=None, help='the path to store samples and models' )

# The basic training setting
parser.add_argument('--batchSize', type=int, default=1, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--envHeight', type=int, default=8, help='the height / width of the input image to network' )
parser.add_argument('--envWidth', type=int, default=16, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Finetuning parameters
parser.add_argument('--nepochBRDF', type=int, default=15, help='the epoch used for testing' )
parser.add_argument('--iterIdVisLamp', type=int, default=119540, help='the iteration used for testing' )
parser.add_argument('--iterIdInvLamp', type=int, default=150000, help='the iteration used for testing' )
parser.add_argument('--iterIdVisWin', type=int, default=120000, help='the iteration used for testing' )
parser.add_argument('--iterIdInvWin', type=int, default=200000, help='the iteration used for testing' )
parser.add_argument('--iterIdDirecIndirec', type=int, default=180000, help='the iteration used for testing' )
parser.add_argument('--iterIdShadow', type=int, default=70000, help='the iteration used for testing')
parser.add_argument('--iterIdPerpixelEnv', type=int, default=240000, help='the iteration used for testing')

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for the size error')
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--winSrcIntWeight', type=float, default=0.001, help='the loss for window light source' )
parser.add_argument('--winSrcAxisWeight', type=float, default=1.0, help='the loss for window light source' )
parser.add_argument('--winSrcLambWeight', type=float, default=0.001, help='the loss for window light source' )

parser.add_argument('--isPredDepth', action='store_true', help='whether to use predicted depth or not')

# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]

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

if opt.experimentPerpixelEnv is None:
    opt.experimentPerpixelEnv = 'check_shadingToLight'
opt.experimentPerpixelEnv = osp.join(curDir, opt.experimentPerpixelEnv )

if opt.testRoot is None:
    opt.testRoot = 'test_full'
    opt.testRoot += '_size%.3f_int%.3f_dir%.3f_lam%.3f_ren%.3f' \
        % (opt.sizeWeight, opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight,
           opt.renderWeight )
    opt.testRoot += '_visWin%d_visLamp%d_invWin%d_invLamp%d' \
        % (opt.iterIdVisWin, opt.iterIdVisLamp, opt.iterIdInvWin, opt.iterIdInvLamp )
    if opt.isPredDepth:
        opt.testRoot += '_predDepth'

opt.testRoot = osp.join(curDir, opt.testRoot )
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )

maxWinNum = 3
maxLampNum  = 7

opt.seed = 32
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

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

# Network for light source prediction
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

renderWindow = renderWindow.renderDirecLighting(sampleNum = 100 )
renderVisLamp = renderVisLamp.renderDirecLighting()
renderInvLamp = renderInvLamp.renderDirecLighting()
renderShadow = renderShadowDepth.renderShadow(
        modelRoot = opt.experimentShadow, iterId = opt.iterIdShadow,
        outputRoot = opt.testRoot)

# Network for direct-indirect lighting predictio
indirectLightNet = modelLight.indirectLightNet()
indirectLightDict = torch.load(
        '{0}/indirectLightNet_iter{1}.pth'.format(opt.experimentDirecIndirec, opt.iterIdDirecIndirec ) )
indirectLightNet.load_state_dict(indirectLightDict['model'] )
for param in indirectLightNet.parameters():
    param.requires_grad = False

lightEncoder = modelLight.encoderLight()
weightDecoder = modelLight.decoderLight(mode = 2 )
axisDecoder = modelLight.decoderLight(mode = 0 )
lambDecoder = modelLight.decoderLight(mode = 1 )

lightEncoderDict = torch.load(
        '{0}/lightEncoder_iter{1}.pth'.format(opt.experimentPerpixelEnv, opt.iterIdPerpixelEnv) )
weightDecoderDict = torch.load(
        '{0}/weightDecoder_iter{1}.pth'.format(opt.experimentPerpixelEnv, opt.iterIdPerpixelEnv ) )
axisDecoderDict = torch.load(
        '{0}/axisDecoder_iter{1}.pth'.format(opt.experimentPerpixelEnv, opt.iterIdPerpixelEnv ) )
lambDecoderDict = torch.load(
        '{0}/lambDecoder_iter{1}.pth'.format(opt.experimentPerpixelEnv, opt.iterIdPerpixelEnv ) )

lightEncoder.load_state_dict(lightEncoderDict['model'] )
weightDecoder.load_state_dict(weightDecoderDict['model'] )
axisDecoder.load_state_dict(axisDecoderDict['model'] )
lambDecoder.load_state_dict(lambDecoderDict['model'] )
for param in lightEncoder.parameters():
    param.requires_grad = False
for param in weightDecoder.parameters():
    param.requires_grad = False
for param in axisDecoder.parameters():
    param.requires_grad = False
for param in lambDecoder.parameters():
    param.requires_grad = False

output2env = models.output2env(envWidth = opt.envWidth, envHeight = opt.envHeight )

# Compute sine weight
theta_val = np.arange(0, opt.envHeight ).astype(np.float32 )
theta_val = (theta_val + 0.5) / opt.envHeight * np.pi / 2.0
phi_val = np.arange(0, opt.envWidth ).astype(np.float32 )
phi_val = (phi_val + 0.5) / opt.envWidth * np.pi * 2.0
phi_val, theta_val = np.meshgrid(phi_val, theta_val )
sinWeight = np.sin(theta_val )
sinWeight = sinWeight.reshape([1, 1, 1, 1, opt.envHeight, opt.envWidth ])
envWeight = torch.from_numpy(sinWeight )
envWeight = envWeight / torch.mean(envWeight )

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

lightEncoder = lightEncoder.cuda()
weightDecoder = weightDecoder.cuda()
axisDecoder = axisDecoder.cuda()
lambDecoder = lambDecoder.cuda()

envWeight = envWeight.cuda()

brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isLightSrc = True, isShading = True, isLight = True,
        phase = 'TEST', isPredDepth = opt.isPredDepth )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 6, shuffle = False, drop_last = False )

j = 0

shadingDirectErrsNpList = np.ones([1, 1], dtype = np.float32 )
shadingErrsNpList = np.ones( [1, 1], dtype = np.float32 )

lightErrsNpList = np.ones( [1, 1], dtype = np.float32 )

nrow = opt.batchSize * 3

testingLog = open('{0}/testingLog.txt'.format(opt.testRoot ), 'w')
for i, dataBatch in enumerate(brdfLoader ):
    j += 1
    # Load brdf
    albedoBatch, normalBatch, roughBatch, \
            depthBatch, depthOrigin, depthScaleBatch, \
            segBRDFBatch, segAllBatch \
            = dataLoader.getBRDF(dataBatch )

    albedoSmallBatch = F.adaptive_avg_pool2d(albedoBatch, [opt.envRow, opt.envCol] )

    # Load image
    im_cpu = dataBatch['im']
    imBatch = im_cpu.cuda()
    imDl_cpu = dataBatch['imDl']
    imDlBatch = imDl_cpu.cuda()
    imDm_cpu = dataBatch['imDm']
    imDmBatch = imDm_cpu.cuda()
    imBatch = torch.cat([imBatch, imDlBatch, imDmBatch], dim=0 )

    # Load envmaps
    env_cpu = dataBatch['envmap']
    envBatch = env_cpu.cuda()
    envDl_cpu = dataBatch['envmapDl']
    envDlBatch = envDl_cpu.cuda()
    envDm_cpu = dataBatch['envmapDm']
    envDmBatch = envDm_cpu.cuda()
    envBatch = torch.cat([envBatch, envDlBatch, envDmBatch], dim=0 )


    semLabel_cpu = dataBatch['semLabel']
    semLabelBatch = semLabel_cpu.cuda()
    semLabelBatch = torch.cat([semLabelBatch, semLabelBatch, semLabelBatch], dim=0 )

    # Load masks
    lightMasks_cpu = dataBatch['lightMasks']
    lightMasksBatch = lightMasks_cpu.cuda()
    lightMasksBatch = torch.cat([lightMasksBatch,
        lightMasksBatch, lightMasksBatch ], dim=0 )

    lightOnMasks_cpu = dataBatch['lightOnMasks']
    lightOnMasksBatch = lightOnMasks_cpu.cuda()
    lightOnMasksDl_cpu = dataBatch['lightOnMasksDl']
    lightOnMasksDlBatch = lightOnMasksDl_cpu.cuda()
    lightOnMasksBatch = torch.cat([lightOnMasksBatch,
        lightOnMasksDlBatch, lightOnMasksBatch ], dim=0 )

    lightMasksSmallBatch = F.adaptive_avg_pool2d(lightMasksBatch, (opt.envRow, opt.envCol ) )
    lightOnMasksSmallBatch = F.adaptive_avg_pool2d(lightOnMasksBatch, (opt.envRow, opt.envCol ) )
    semLabelSmallBatch = F.adaptive_avg_pool2d(semLabelBatch, (opt.envRow, opt.envCol ) )
    semLabelSmallBatch = (semLabelSmallBatch > 0.999).float()

    # Load visible window
    visWinNum, winMasksBatch, onWinMasksBatch, \
            visWinPlanesBatch, visWinSrcsBatch, \
            visWinShadingsBatch, visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
            visWinShadowsBatch \
            = dataLoader.getVisWindow(dataBatch )

    # Load visible lamp
    visLampNum, lampMasksBatch, onLampMasksBatch, \
            visLampCentersBatch, visLampAxesBatch, \
            visLampShadingsBatch, visLampShadingsNoBatch, \
            visLampShadowsBatch \
            = dataLoader.getVisLamp(dataBatch )

    # Load invisible window
    invWinNum, invWinOnBatch, \
            invWinPlanesBatch, invWinSrcsBatch, \
            invWinShadingsBatch, invWinShadingsNoBatch, invWinShadingsNoAppBatch, \
            invWinShadowsBatch \
            = dataLoader.getInvWindow(dataBatch )

    # Load invisible lamp
    invLampNum, invLampOnBatch, \
            invLampCentersBatch, invLampAxesBatch, \
            invLampShadingsBatch, \
            invLampShadingsNoBatch, invLampShadowsBatch \
            = dataLoader.getInvLamp(dataBatch )

    # Load shading
    shadingBatch, shadingDirectBatch \
            = dataLoader.getShading(dataBatch )

    nameBatch = dataBatch['name']
    batchSize = imBatch.size(0 )
    for m in range(0, int(batchSize / 3.0 ) ):
        print('%s visWin %d invWin %d visLamp %d invLamp %d' \
                % (nameBatch[m], visWinNum[m], invWinNum[m], visLampNum[m], invLampNum[m] ) )
        testingLog.write('%s visWin %d invWin %d visLamp %d invLamp %d\n' \
                % (nameBatch[m], visWinNum[m], invWinNum[m], visLampNum[m], invLampNum[m] ) )

    depthMax = torch.max(torch.max(depthBatch, dim=2, keepdim=True )[0], dim=3, keepdim=True )[0]
    depthBatch = depthBatch * segAllBatch + (1 - segAllBatch ) * depthMax
    inputBatch = torch.cat([imBatch, depthBatch], dim=1 )

    # Predict the large BRDF
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred, _ = albedoDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth ] )
    normalPred, _ = normalDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth] )
    roughPred, _ = roughDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth] )

    # Down sample the image and masks
    winMasksSmallBatch = F.adaptive_avg_pool2d(winMasksBatch, (opt.envRow, opt.envCol ) )
    onWinMasksSmallBatch = F.adaptive_avg_pool2d(onWinMasksBatch, (opt.envRow, opt.envCol ) )

    lampMasksSmallBatch = F.adaptive_avg_pool2d(lampMasksBatch, (opt.envRow, opt.envCol ) )
    onLampMasksSmallBatch = F.adaptive_avg_pool2d(onLampMasksBatch, (opt.envRow, opt.envCol ) )

    segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )
    imSmallBatch = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol ) )

    albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
    normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
    roughDS = F.adaptive_avg_pool2d(roughPred, (opt.envRow, opt.envCol ) )
    depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

    # Predict the visible window
    visWinCenterPreds = []
    visWinNormalPreds = []
    visWinYAxisPreds = []
    visWinXAxisPreds = []

    visWinSrcPreds = []
    visWinSrcSkyPreds = []
    visWinSrcGrdPreds = []

    visWinOns = torch.clamp( torch.sum(torch.sum(
                onWinMasksSmallBatch, dim=2 ), dim = 2 ), 0, 1 )
    visWinOns = visWinOns.cpu().numpy()
    onVisWinNum = np.sum(visWinOns )

    if onVisWinNum > 0:
        for m in range(0, batchSize ):
            for n in range(0, maxWinNum ):
                if visWinOns[m, n] == 1:
                    visWinCenterPred, visWinNormalPred, \
                            visWinYAxisPred, visWinXAxisPred, \
                            visWinSrcPred, visWinSrcSkyPred, visWinSrcGrdPred \
                            = visWinNet(
                                imSmallBatch[m:m+1, :],
                                depthDS.detach()[m:m+1, :],
                                albedoDS.detach()[m:m+1, :],
                                lightOnMasksSmallBatch[m:m+1, :],
                                onWinMasksSmallBatch[m:m+1, n:n+1]
                            )

                    visWinSrcPreds.append(visWinSrcPred )
                    visWinSrcSkyPreds.append(visWinSrcSkyPred )
                    visWinSrcGrdPreds.append(visWinSrcGrdPred )

                    visWinCenterPreds.append(visWinCenterPred )
                    visWinNormalPreds.append(visWinNormalPred )
                    visWinYAxisPreds.append(visWinYAxisPred )
                    visWinXAxisPreds.append(visWinXAxisPred )
                else:
                    visWinCenterPreds.append(None )
                    visWinNormalPreds.append(None )
                    visWinYAxisPreds.append(None )
                    visWinXAxisPreds.append(None )

                    visWinSrcPreds.append(None )
                    visWinSrcSkyPreds.append(None )
                    visWinSrcGrdPreds.append(None )

    # Predict the visible lamp
    visLampCenterPreds = []
    visLampSrcPreds = []
    onVisLampNum = 0

    visLampOns = torch.clamp( torch.sum(torch.sum(
                (onLampMasksSmallBatch == 1).float(), dim=2 ), dim = 2 ), 0, 1 )
    visLampOns = visLampOns.cpu().numpy()
    onVisLampNum = np.sum(visLampOns )

    if onVisLampNum > 0:
        for m in range(0, batchSize ):
            for n in range(0, maxLampNum ):
                if visLampOns[m, n] == 1:
                    visLampCenterPred, visLampSrcPred \
                            = visLampNet(
                                imSmallBatch[m:m+1, :],
                                depthDS.detach()[m:m+1, :],
                                albedoDS.detach()[m:m+1, :],
                                lightOnMasksSmallBatch[m:m+1, :],
                                onLampMasksSmallBatch[m:m+1, n:n+1, :]
                            )
                    visLampCenterPreds.append(visLampCenterPred )
                    visLampSrcPreds.append(visLampSrcPred )
                else:
                    visLampCenterPreds.append(None )
                    visLampSrcPreds.append(None )


    # Predict the invisible window
    onInvWinNum = 0
    invWinCenterPred, invWinNormalPred, \
            invWinYAxisPred, invWinXAxisPred, \
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred \
            = invWinNet(
                    imSmallBatch,
                    depthDS.detach(), albedoDS.detach(),
                    lightOnMasksSmallBatch
            )

    # Predict invisible lamp
    onInvLampNum = 0
    invLampAxesPred, invLampCenterPred, invLampSrcPred \
            = invLampNet(
                    imSmallBatch,
                    depthDS.detach(), albedoDS.detach(),
                    lightOnMasksSmallBatch
            )

    # Compute shadows
    visWinShadowInits, visWinShadowPreds, \
        visLampShadowInits, visLampShadowPreds, \
        invWinShadowInit, invWinShadowPred, \
        invLampShadowInit, invLampShadowPred \
        = renderShadow.forward(
            depthDS, normalDS,
            visWinCenterPreds, visWinXAxisPreds, visWinYAxisPreds, onWinMasksSmallBatch,
            visLampCenterPreds, onLampMasksSmallBatch,
            invWinCenterPred, invWinXAxisPred, invWinYAxisPred,
            invLampAxesPred, invLampCenterPred,
            objName=None )

    # Compute rendering error for visible window
    visWinShadingPreds, visWinShadingNoPreds \
        = renderVisWindowArr(
            visWinCenterPreds, visWinNormalPreds, visWinXAxisPreds, visWinYAxisPreds,
            visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
            depthDS, normalDS, albedoDS, albedoSmallBatch,
            visWinShadowPreds,
            segEnvBatch,
            visWinOns, onVisWinNum, batchSize, maxWinNum
        )


    # Compute rendering error for visible lamp
    visLampShadingPreds, visLampShadingNoPreds, visLampPointsPreds \
        = renderVisLampArr(
            visLampCenterPreds, visLampSrcPreds, onLampMasksSmallBatch,
            depthDS, normalDS, albedoDS, albedoSmallBatch,
            visLampShadowPreds,
            segEnvBatch,
            visLampOns, onVisLampNum, batchSize, maxLampNum
        )

    # Compute rendering error for invisible window
    invWinShadingPred, invWinShadingNoPred \
        = renderInvWindowArr(
            invWinCenterPred, invWinNormalPred, invWinXAxisPred, invWinYAxisPred,
            invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
            depthDS, normalDS, albedoDS, albedoSmallBatch,
            invWinShadowPred,
            segEnvBatch
        )

    # Compute rendering error for invisible lamp
    invLampShadingPred, invLampShadingNoPred \
        = renderInvLampArr(
            invLampAxesPred, invLampCenterPred,
            invLampSrcPred,
            depthDS, normalDS, albedoDS, albedoSmallBatch,
            invLampShadowPred,
            segEnvBatch
        )

    # Predict the global illumination
    if onVisWinNum > 0:
        visWinShadingNoPreds = torch.cat(visWinShadingNoPreds, dim=0 ).reshape( batchSize, maxWinNum, 3, opt.envRow, opt.envCol )
        visWinShadingPreds = torch.cat(visWinShadingPreds, dim=0 ).reshape( batchSize, maxWinNum, 3, opt.envRow, opt.envCol )

    if onVisLampNum > 0:
        visLampShadingNoPreds = torch.cat(visLampShadingNoPreds, dim=0 ).reshape(batchSize, maxLampNum, 3, opt.envRow, opt.envCol )
        visLampShadingPreds = torch.cat(visLampShadingPreds, dim=0 ).reshape(batchSize, maxLampNum, 3, opt.envRow, opt.envCol )

    shadingDirectPred = invWinShadingPred + invLampShadingPred
    if onVisWinNum > 0:
        shadingDirectPred += torch.sum(visWinShadingPreds, dim=1 )

    if onVisLampNum > 0:
        shadingDirectPred += torch.sum(visLampShadingPreds, dim=1 )

    shadingDirectPredInput = torch.atan(shadingDirectPred ) / np.pi * 2.0

    shadingIndirectPred = indirectLightNet(
            albedoDS.detach(),
            normalDS.detach(),
            depthDS.detach(),
            shadingDirectPredInput,
            lightOnMasksSmallBatch )

    shadingPred = shadingIndirectPred + shadingDirectPred

    renderedPred = shadingPred * albedoDS
    renderedGt = shadingBatch * albedoSmallBatch

    shgDirectErr = torch.mean(torch.abs(shadingDirectPred - shadingDirectBatch ) * segEnvBatch )

    shgErr = torch.mean( torch.abs(shadingPred - shadingBatch ) * segEnvBatch )

    # Predict per-pixel lighting
    inputBatch = torch.cat([albedoDS, normalDS, roughDS, depthDS, lightOnMasksSmallBatch ], dim=1 )

    x1, x2, x3, x4, x5, x6 = lightEncoder(
        inputBatch.detach(),
        shadingBatch.detach() )

    # Prediction
    axisPred = axisDecoder(x1, x2, x3, x4, x5, x6, envBatch )
    lambPred = lambDecoder(x1, x2, x3, x4, x5, x6, envBatch )
    weightPred = weightDecoder(x1, x2, x3, x4, x5, x6, envBatch )
    bn, SGNum, _, envRow, envCol = axisPred.size()

    # Compute the recontructed error
    envPred, axisPred, lambPred, weightPred \
        = output2env.output2env(axisPred, lambPred, weightPred )

    lightErr = torch.mean(
        torch.pow(torch.log(envPred + 1 ) - torch.log(envBatch + 1 ),
                  2 ) * segEnvBatch.unsqueeze(-1).unsqueeze(-1) * envWeight )

    # Write shading errors
    utils.writeErrToScreen('shgDirect', [shgDirectErr ], opt.iterIdDirecIndirec, j )
    utils.writeErrToFile('shgDirect', [shgDirectErr ], testingLog, opt.iterIdDirecIndirec, j )
    utils.writeErrToScreen('shg',[shgErr], opt.iterIdDirecIndirec, j )
    utils.writeErrToFile('shg', [shgErr], testingLog, opt.iterIdDirecIndirec, j )

    shadingErrsNpList = np.concatenate( [shadingErrsNpList, utils.turnErrorIntoNumpy( [shgErr] ) ], axis=0 )
    shadingDirectErrsNpList = np.concatenate( [shadingDirectErrsNpList, utils.turnErrorIntoNumpy( [shgDirectErr ] ) ], axis=0 )

    utils.writeNpErrToScreen('shadingDirectAccu', np.mean(shadingDirectErrsNpList[1:j+1, :], axis=0), opt.iterIdDirecIndirec, j )
    utils.writeNpErrToFile('shadingDirectAccu', np.mean(shadingDirectErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterIdDirecIndirec, j )
    utils.writeNpErrToScreen('shadingAccu', np.mean(shadingErrsNpList[1:j+1, :], axis=0), opt.iterIdDirecIndirec, j )
    utils.writeNpErrToFile('shadingAccu', np.mean(shadingErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterIdDirecIndirec, j )

    # Write per-pixel lighting errors
    utils.writeErrToScreen('light', [lightErr ], opt.iterIdPerpixelEnv, j )
    utils.writeErrToFile('light', [lightErr ], testingLog, opt.iterIdPerpixelEnv, j )

    lightErrsNpList = np.concatenate( [ lightErrsNpList, utils.turnErrorIntoNumpy( [lightErr ] ) ], axis=0 )

    utils.writeNpErrToScreen('lightAccu', np.mean(lightErrsNpList[1:j+1, :], axis=0), opt.iterIdPerpixelEnv, j )
    utils.writeNpErrToFile('lightAccu', np.mean(lightErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterIdPerpixelEnv, j )

    if j == 1 or j % 200 == 0:
        # Save the groundtruth results
        vutils.save_image( albedoBatch ** (1.0/2.2), '{0}/{1}_albedoGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( 0.5*(normalBatch + 1), '{0}/{1}_normalGt.png'.format(opt.testRoot, j), nrow=nrow )
        depthOutGt = 1 / torch.clamp(depthBatch + 1, 1e-6, 10 )
        vutils.save_image( depthOutGt, '{0}/{1}_depthGt.png'.format(opt.testRoot, j), nrow=nrow )

        # Save the predicted BRDF
        vutils.save_image( albedoPred ** (1.0/2.2), '{0}/{1}_albedoPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( 0.5*(normalPred + 1), '{0}/{1}_normalPred.png'.format(opt.testRoot, j), nrow=nrow )

        # Output Ground-truth point clouds
        normalSmallBatch = F.adaptive_avg_pool2d(normalBatch, [opt.envRow, opt.envCol ] )
        utils.writeDepthAsPointClouds(
                depthDS,
                normalSmallBatch,
                segEnvBatch,
                '{0}/{1}_roomGt.ply'.format(opt.testRoot, j) )

        # Output the predicted point clouds
        utils.writeDepthAsPointClouds(
                depthDS,
                normalDS,
                segEnvBatch,
                '{0}/{1}_roomPred.ply'.format(opt.testRoot, j) )

        # Output Ground-truth window
        if onVisWinNum > 0:
            utils.writeWindowList(
                    visWinCenterPreds,
                    visWinYAxisPreds,
                    visWinXAxisPreds,
                    maxWinNum,
                    '{0}/{1}_visWinPred.obj'.format(opt.testRoot, j ) )

            vutils.save_image( (visWinShadingPreds ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visWinShadingPred.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( (visWinShadingNoPreds ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visWinShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

            vutils.save_image( visWinShadowPreds.transpose(0, 1).reshape( batchSize * maxWinNum, 1, opt.envRow, opt.envCol ),
                '{0}/{1}_visWinShadowPred.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( visWinShadowInits.transpose(0, 1).reshape( batchSize * maxWinNum, 1, opt.envRow, opt.envCol ),
                '{0}/{1}_visWinShadowInit.png'.format(opt.testRoot, j), nrow=nrow )

            vutils.save_image( winMasksBatch.transpose(0, 1).reshape(batchSize * maxWinNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visWinMask.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( onWinMasksBatch.transpose(0, 1).reshape(batchSize * maxWinNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visOnWinMask.png'.format(opt.testRoot, j), nrow=nrow )

        # Output Ground-truth lamp
        if onVisLampNum > 0:
            utils.writeLampList(
                    visLampCenterPreds,
                    depthDS,
                    normalDS,
                    onLampMasksSmallBatch,
                    maxLampNum,
                    '{0}/{1}_visLampPred.ply'.format(opt.testRoot, j) )

            vutils.save_image( (visLampShadingPreds ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadingPred.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( (visLampShadingNoPreds ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

            vutils.save_image( visLampShadowPreds.transpose(0, 1).reshape( batchSize * maxLampNum, 1, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadowPred.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( visLampShadowInits.transpose(0, 1).reshape( batchSize * maxLampNum, 1, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadowInit.png'.format(opt.testRoot, j), nrow=nrow )

            vutils.save_image( lampMasksBatch.transpose(0, 1).reshape(batchSize * maxLampNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visLampMask.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( onLampMasksBatch.transpose(0, 1).reshape(batchSize * maxLampNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visOnLampMask.png'.format(opt.testRoot, j), nrow=nrow )

        # Output Ground-truth window
        vutils.save_image( invWinShadingNoPred ** (1.0/2.2), '{0}/{1}_invWinShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadingPred ** (1.0/2.2), '{0}/{1}_invWinShadingPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadowPred, '{0}/{1}_invWinShadowPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadowInit, '{0}/{1}_invWinShadowInit.png'.format(opt.testRoot, j), nrow=nrow )

        utils.writeWindowBatch(
                invWinCenterPred.unsqueeze(1),
                invWinYAxisPred.unsqueeze(1),
                invWinXAxisPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                '{0}/{1}_invWinPred.obj'.format(opt.testRoot, j) )

        # Output Ground-truth lamp
        vutils.save_image( (invLampShadingNoPred ** (1.0/2.2) ), '{0}/{1}_invLampShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( invLampShadowPred, '{0}/{1}_invLampShadowPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invLampShadowInit, '{0}/{1}_invLampShadowInit.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invLampShadingPred ** (1.0/2.2) ), '{0}/{1}_invLampShadingPred.png'.format(opt.testRoot, j), nrow=nrow )

        utils.writeLampBatch(
                invLampAxesPred.unsqueeze(1),
                invLampCenterPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                '{0}/{1}_invLampPred.ply'.format(opt.testRoot, j), )

        vutils.save_image( ((shadingBatch)**(1.0/2.2) ).data,
            '{0}/{1}_shadingGt.png'.format(opt.testRoot, j), nrow = nrow )
        vutils.save_image( ((shadingDirectBatch )**(1.0/2.2) ).data,
            '{0}/{1}_shadingDirectGt.png'.format(opt.testRoot, j), nrow = nrow )

        vutils.save_image( ((renderedGt)**(1.0/2.2) ).data,
            '{0}/{1}_renderedGt.png'.format(opt.testRoot, j), nrow = nrow )

        vutils.save_image( ( (shadingPred  )**(1.0/2.2) ).data,
            '{0}/{1}_shadingPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( ( (renderedPred  )**(1.0/2.2) ).data,
            '{0}/{1}_renderedPred.png'.format(opt.testRoot, j), nrow=nrow )

        # Output the groundtruth lighting and image
        vutils.save_image( (imBatch )**(1.0/2.2), '{0}/{1}_im.png'.format(opt.testRoot, j ), nrow=nrow )
        vutils.save_image( semLabelBatch, '{0}/{1}_semLabel.png'.format(opt.testRoot, j ), nrow=nrow )

        utils.writeEnvToFile(envBatch, nrow, '{0}/{1}_envmapGt.hdr'.format(opt.testRoot, j) )
        utils.writeEnvToFile(envPred, nrow, '{0}/{1}_envmapPred.hdr'.format(opt.testRoot, j ) )

testingLog.close()

# Save the error record
np.save('{0}/shadingDirectErr_{1}.npy'.format(opt.testRoot, opt.iterIdDirecIndirec), shadingDirectErrsNpList )
np.save('{0}/shadingErr_{1}.npy'.format(opt.testRoot, opt.iterIdDirecIndirec), shadingErrsNpList )
np.save('{0}/lightError_{1}.npy'.format(opt.testRoot, opt.iterIdPerpixelEnv ), lightErrsNpList )
