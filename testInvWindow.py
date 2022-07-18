import utils
import torch
import numpy as np
import argparse
import random
import os
import models
import modelLight
import renderWindow
import torchvision.utils as vutils
import dataLoaderInvWindow as dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
import lossFunctions

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images')
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--testRoot', default=None, help='the path to store testing samples' )
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=50, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=3, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Finetuning parameters
parser.add_argument('--iterId', type=int, default=200000, help='the iteration used for fine-tuning' )

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for the size error')
parser.add_argument('--winSrcIntWeight', type=float, default=0.001, help='the loss for window light source' )
parser.add_argument('--winSrcAxisWeight', type=float, default=1.0, help='the loss for window light source' )
parser.add_argument('--winSrcLambWeight', type=float, default=0.001, help='the loss for window light source' )

parser.add_argument('--isPredDepth', action='store_true', help='whether to use predicted depth or not')

# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_invWindow'
    opt.experiment += '_bn%d_shg%.3f_geo%.3f_size%.3f_ren%.3f_srcInt%.3f_srcAxis%.3f_arcLamb%.3f' \
        % (opt.batchSize, opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight,
           opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight )

opt.experiment = osp.join(curDir, opt.experiment )
if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check_', 'test_')
    if opt.isPredDepth:
        opt.testRoot += '_predDepth'
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )

maxWinNum = 3

opt.seed = opt.iterId + 16
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

# Network for lighting prediction
invWinNet = modelLight.windowNet(isInv = True )

invWinDict = torch.load('{0}/invWinNet_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
invWinNet.load_state_dict(invWinDict['model'] )
for param in invWinNet.parameters():
    param.requires_grad = False

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

encoder.load_state_dict(torch.load('{0}/encoder_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
albedoDecoder.load_state_dict(torch.load('{0}/albedo_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
normalDecoder.load_state_dict(torch.load('{0}/normal_{1}.pth'.format(
    opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in encoder.parameters():
    param.requires_grad = False
for param in albedoDecoder.parameters():
    param.requires_grad = False
for param in normalDecoder.parameters():
    param.requires_grad = False

encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )

invWinNet = nn.DataParallel(invWinNet, device_ids = opt.deviceIds )

renderWindow = renderWindow.renderDirecLighting(sampleNum = 100  )

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
roughDecoder = roughDecoder.cuda()
normalDecoder = normalDecoder.cuda()

invWinNet = invWinNet.cuda()

brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isLightSrc = True, phase = 'TEST', isPredDepth = opt.isPredDepth )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 0, shuffle = False, drop_last = False )

j = 0

invWinPointsErrsNpList = np.ones([1, 1], dtype = np.float32 )
invWinPointsErrsNpList = np.ones([1, 1], dtype = np.float32 )
invWinNormalErrsNpList = np.ones([1, 1], dtype=np.float32 )
invWinSizeErrsNpList = np.ones([1, 1], dtype=np.float32 )

invWinSrcErrsNpList = np.ones([1, 3], dtype = np.float32 )
invWinSrcSkyErrsNpList = np.ones([1, 3], dtype = np.float32 )
invWinSrcGrdErrsNpList = np.ones([1, 3], dtype = np.float32 )

invWinShgErrsNpList = np.ones( [1, 1], dtype = np.float32 )
invWinShgAppErrsNpList = np.ones( [1, 1], dtype = np.float32 )
invWinRenErrsNpList = np.ones( [1, 1], dtype = np.float32 )

invWinNumNpList = np.ones([1, 1], dtype=np.float32 )

nrow = opt.batchSize * 3

testingLog = open('{0}/testingLog_iter{1}.txt'.format(opt.testRoot, opt.iterId ), 'w')
for i, dataBatch in enumerate(brdfLoader ):
    j += 1
    # Load brdf
    albedoBatch, normalBatch, roughBatch, \
            depthBatch, depthOriginBatch, depthScaleBatch, \
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

    # Load invisible window
    invWinNum, invWinOnBatch, \
            invWinPlanesBatch, invWinSrcsBatch, \
            invWinShadingsBatch, invWinShadingsNoBatch, invWinShadingsNoAppBatch, \
            invWinShadowsBatch \
            = dataLoader.getInvWindow(dataBatch )

    nameBatch = dataBatch['name']
    batchSize = imBatch.size(0 )
    for m in range(0, int(batchSize / 3.0 ) ):
        print('%s %d' % (nameBatch[m], invWinNum[m] ) )
        testingLog.write('%s %d\n' % (nameBatch[m], invWinNum[m] ) )

    onInvWinNum = 0

    invWinSrcsGt, invWinPlanesGt, invWinOnGt = [], [], []
    invWinShadingGt, invWinShadingNoGt, invWinShadingNoApp, invWinShadowGt = [], [], [], []
    for m in range(0, batchSize ):
        maxInt = -1
        winId = 0
        for n in range(0, maxWinNum ):
            if invWinOnBatch[m, n] > 0:
                winInt = torch.mean(invWinShadingsBatch[m, n] )
                winInt = winInt.detach().cpu().item()
                if winInt > maxInt:
                    maxInt = winInt
                    winId = n

        if maxInt < 0.01:
            invWinOnGt.append(0 )
            invWinShadingGt.append(torch.zeros(1, 3, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
            invWinShadingNoGt.append(torch.zeros(1, 3, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
            invWinShadowGt.append(torch.ones(1, 1, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
            invWinShadingNoApp.append(torch.zeros(1, 3, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
        else:
            onInvWinNum += 1
            invWinOnGt.append(1 )
            invWinShadingGt.append(invWinShadingsNoBatch[m:m+1, winId, :] * invWinShadowsBatch[m:m+1, winId, :] )
            invWinShadingNoGt.append(invWinShadingsNoBatch[m:m+1, winId, :] )
            invWinShadowGt.append(invWinShadowsBatch[m:m+1, winId, :] )
            invWinShadingNoApp.append(invWinShadingsNoAppBatch[m:m+1, winId, :] )

        invWinSrcsGt.append(invWinSrcsBatch[m:m+1, winId ] )
        invWinPlanesGt.append(invWinPlanesBatch[m:m+1, winId ] )

    invWinSrcsGt = torch.cat(invWinSrcsGt, dim=0 )
    invWinPlanesGt = torch.cat(invWinPlanesGt, dim=0 )

    invWinShadingGt = torch.cat(invWinShadingGt, dim=0 )
    invWinShadingNoGt = torch.cat(invWinShadingNoGt, dim=0 )
    invWinShadingNoApp = torch.cat(invWinShadingNoApp, dim=0 )
    invWinShadowGt = torch.cat(invWinShadowGt, dim=0 )

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

    segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )
    imSmallBatch = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol ) )

    albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
    normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
    depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

    # Predict the direct lighting of invisible window
    invWinCenterPred, invWinNormalPred, \
            invWinYAxisPred, invWinXAxisPred, \
            invWinSrcSunPred, invWinSrcSkyPred, invWinSrcGrdPred \
            = invWinNet(
                    imSmallBatch,
                    depthDS.detach(), albedoDS.detach(),
                    lightOnMasksSmallBatch
            )
    invWinShadingNoPred = renderWindow.forward(
                    invWinCenterPred,
                    invWinNormalPred,
                    invWinXAxisPred,
                    invWinYAxisPred,
                    invWinSrcSunPred,
                    invWinSrcSkyPred,
                    invWinSrcGrdPred,
                    depthDS.detach(),
                    normalDS.detach() )

    invWinPointsErr, invWinNormalErr, invWinSizeErr, \
            invWinSrcErr, invWinSrcSkyErr, invWinSrcGrdErr, \
            = lossFunctions.invWindowLoss(
                    invWinCenterPred, invWinNormalPred,
                    invWinXAxisPred, invWinYAxisPred,
                    invWinSrcSunPred, invWinSrcSkyPred, invWinSrcGrdPred,
                    invWinOnGt,
                    invWinPlanesGt, invWinSrcsGt,
                    depthScaleBatch, isTest = True )
    invWinShadingPred = invWinShadingNoPred * invWinShadowGt
    invWinShadingApp = invWinShadingNoApp * invWinShadowGt
    invWinShgErr = torch.mean( torch.abs(invWinShadingNoPred - invWinShadingNoGt ) * segEnvBatch )
    invWinShgAppErr = torch.mean( torch.abs(invWinShadingNoApp - invWinShadingNoGt ) * segEnvBatch )

    invWinRenderGt = invWinShadingNoGt * albedoSmallBatch
    invWinRenderPred = invWinShadingNoPred * albedoDS
    invWinRenErr = torch.mean( torch.abs(invWinRenderPred - invWinRenderGt ) * segEnvBatch )

    invWinSrcErrMean, invWinSrcSkyErrMean, invWinSrcGrdErrMean = [], [], []
    for n in range(0, 3):
        invWinSrcErrMean.append(invWinSrcErr[n] / max(onInvWinNum, 1 ) )
        invWinSrcSkyErrMean.append(invWinSrcSkyErr[n] / max(onInvWinNum, 1 ) )
        invWinSrcGrdErrMean.append(invWinSrcGrdErr[n] / max(onInvWinNum, 1 ) )

    # Write errors to screen
    utils.writeErrToScreen('invWinPoints', [invWinPointsErr / max(onInvWinNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('invWinNormal', [invWinNormalErr / max(onInvWinNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('invWinSize', [invWinSizeErr / max(onInvWinNum, 1) ], opt.iterId, j )

    utils.writeErrToScreen('invWinSrc', invWinSrcErrMean, opt.iterId, j )
    utils.writeErrToScreen('invWinSrcSky', invWinSrcSkyErrMean, opt.iterId, j )
    utils.writeErrToScreen('invWinSrcGrd', invWinSrcGrdErrMean, opt.iterId, j )

    utils.writeErrToScreen('invWinShading', [invWinShgErr ], opt.iterId, j )
    utils.writeErrToScreen('invWinShadingApp', [invWinShgAppErr ], opt.iterId, j )
    utils.writeErrToScreen('invWinRender', [invWinRenErr ], opt.iterId, j )

    # Write errors to file
    utils.writeErrToFile('invWinPoints', [invWinPointsErr / max(onInvWinNum, 1 ) ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinNormal', [invWinNormalErr / max(onInvWinNum, 1 ) ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinSize', [invWinSizeErr / max(onInvWinNum, 1 ) ], testingLog, opt.iterId, j )

    utils.writeErrToFile('invWinSrc', invWinSrcErrMean, testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinSrcSky', invWinSrcSkyErrMean, testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinSrcGrd', invWinSrcGrdErrMean, testingLog, opt.iterId, j )

    utils.writeErrToFile('invWinShading', [invWinShgErr ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinShadingApp', [invWinShgAppErr ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invWinRender', [invWinRenErr ], testingLog, opt.iterId, j )

    # Accumulate errors
    invWinPointsErrsNpList = np.concatenate( [invWinPointsErrsNpList, utils.turnErrorIntoNumpy( [invWinPointsErr] )], axis=0 )
    invWinNormalErrsNpList = np.concatenate( [invWinNormalErrsNpList, utils.turnErrorIntoNumpy( [invWinNormalErr] )], axis=0 )
    invWinSizeErrsNpList = np.concatenate( [invWinSizeErrsNpList, utils.turnErrorIntoNumpy( [invWinSizeErr] )], axis=0 )

    invWinSrcErrsNpList = np.concatenate( [invWinSrcErrsNpList, utils.turnErrorIntoNumpy( invWinSrcErr )], axis=0 )
    invWinSrcSkyErrsNpList = np.concatenate( [invWinSrcSkyErrsNpList, utils.turnErrorIntoNumpy( invWinSrcSkyErr )], axis=0 )
    invWinSrcGrdErrsNpList = np.concatenate( [invWinSrcGrdErrsNpList, utils.turnErrorIntoNumpy( invWinSrcGrdErr )], axis=0 )

    invWinShgErrsNpList = np.concatenate( [invWinShgErrsNpList, utils.turnErrorIntoNumpy( [invWinShgErr] ) ], axis=0 )
    invWinShgAppErrsNpList = np.concatenate( [invWinShgAppErrsNpList, utils.turnErrorIntoNumpy( [invWinShgAppErr] ) ], axis=0 )
    invWinRenErrsNpList = np.concatenate( [invWinRenErrsNpList, utils.turnErrorIntoNumpy( [invWinRenErr] ) ], axis=0 )

    invWinNumNpList = np.concatenate([invWinNumNpList, utils.turnErrorIntoNumpy([onInvWinNum ] ) ], axis=0 )

    torch.cuda.empty_cache()

    # Write errors to screen
    invWinNumSum = max(np.sum(invWinNumNpList[1 : j+1] ), 1 )

    utils.writeNpErrToScreen('invWinPointsAccu', np.sum(invWinPointsErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invWinNormalAccu', np.sum(invWinNormalErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invWinSizeAccu', np.sum(invWinSizeErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )

    utils.writeNpErrToScreen('invWinSrcAccu', np.sum(invWinSrcErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invWinSrcSkyAccu', np.sum(invWinSrcSkyErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invWinSrcGrdAccu', np.sum(invWinSrcGrdErrsNpList[1:j+1, :], axis=0) / invWinNumSum, opt.iterId, j )

    utils.writeNpErrToScreen('invWinShgAccu', np.mean(invWinShgErrsNpList[1:j+1, :], axis=0), opt.iterId, j )
    utils.writeNpErrToScreen('invWinShgAppAccu', np.mean(invWinShgAppErrsNpList[1:j+1, :], axis=0), opt.iterId, j )
    utils.writeNpErrToScreen('invWinRenAccu', np.mean(invWinRenErrsNpList[1:j+1, :], axis=0), opt.iterId, j )

    # Write errors to file
    utils.writeNpErrToFile('invWinPointsAccu', np.sum(invWinPointsErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinNormalAccu', np.sum(invWinNormalErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinSizeAccu', np.sum(invWinSizeErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )

    utils.writeNpErrToFile('invWinSrcAccu', np.sum(invWinSrcErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinSrcSkyAccu', np.sum(invWinSrcSkyErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinSrcGrdAccu', np.sum(invWinSrcGrdErrsNpList[1:j+1, :], axis=0) / invWinNumSum, testingLog, opt.iterId, j )

    utils.writeNpErrToFile('invWinShgAccu', np.mean(invWinShgErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinShgAppAccu', np.mean(invWinShgAppErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invWinRenAccu', np.mean(invWinRenErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )

    if j == 1 or j% 200 == 0:
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
        utils.writeWindowBatch(
                invWinPlanesBatch[:, :, 0:3],
                invWinPlanesBatch[:, :, 6:9],
                invWinPlanesBatch[:, :, 9:12],
                invWinOnBatch.detach().cpu().numpy(),
                maxWinNum,
                '{0}/{1}_invWinGt.obj'.format(opt.testRoot, j) )

        vutils.save_image( (invWinShadingsNoBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
            '{0}/{1}_invWinShadingNoGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invWinShadingsNoAppBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
            '{0}/{1}_invWinShadingNoApp.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invWinShadingsBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
            '{0}/{1}_invWinShadingGt.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( invWinShadingNoGt ** (1.0/2.2), '{0}/{1}_invWinShadingNoGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadingNoApp ** (1.0/2.2), '{0}/{1}_invWinShadingNoAppSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadingNoPred ** (1.0/2.2), '{0}/{1}_invWinShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( invWinShadingGt ** (1.0/2.2), '{0}/{1}_invWinShadingGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinShadingPred ** (1.0/2.2), '{0}/{1}_invWinShadingPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( invWinRenderGt ** (1.0/2.2), '{0}/{1}_invWinRenderGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( invWinRenderPred ** (1.0/2.2), '{0}/{1}_invWinRenderPred.png'.format(opt.testRoot, j), nrow=nrow )

        utils.writeWindowBatch(
                invWinCenterPred.unsqueeze(1),
                invWinYAxisPred.unsqueeze(1),
                invWinXAxisPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                '{0}/{1}_invWinPred.obj'.format(opt.testRoot, j) )

        # Output the groundtruth lighting and image
        vutils.save_image( (imBatch )**(1.0/2.2), '{0}/{1}_im.png'.format(opt.testRoot, j ), nrow=nrow )

        vutils.save_image( semLabelBatch, '{0}/{1}_semLabel.png'.format(opt.testRoot, j ), nrow=nrow )

testingLog.close()

# Save the error record
np.save('{0}/invWinPointsError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinPointsErrsNpList )
np.save('{0}/invWinNormalError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinNormalErrsNpList )
np.save('{0}/invWinSizeError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinSizeErrsNpList )

np.save('{0}/invWinSrcError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinSrcErrsNpList )
np.save('{0}/invWinSrcSkyError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinSrcSkyErrsNpList )
np.save('{0}/invWinSrcGrdError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinSrcGrdErrsNpList )

np.save('{0}/invWinShgError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinShgErrsNpList )
np.save('{0}/invWinShgAppError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinShgAppErrsNpList )
np.save('{0}/invWinRenError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinRenErrsNpList )

np.save('{0}/invWinNum_iter{1}.npy'.format(opt.testRoot, opt.iterId), invWinNumNpList )
