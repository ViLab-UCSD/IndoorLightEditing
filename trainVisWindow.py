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
import torchvision.utils as vutils
import dataLoaderVisWindow as dataLoader
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
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=10, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=3, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Finetuning parameters
parser.add_argument('--isFineTune', action='store_true', help='fine tune the network for global local lighting prediction' )
parser.add_argument('--iterId', type=int, default=0, help='the iteration used for fine-tuning' )

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for size error')
parser.add_argument('--winSrcIntWeight', type=float, default=0.001, help='the loss for window light source' )
parser.add_argument('--winSrcAxisWeight', type=float, default=1.0, help='the loss for window light source' )
parser.add_argument('--winSrcLambWeight', type=float, default=0.001, help='the loss for window light source' )
# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_visWindow'
    opt.experiment += '_bn%d_shg%.3f_geo%.3f_size%.3f_ren%.3f_srcInt%.3f_srcAxis%.3f_srcLamb%.3f' \
        % (opt.batchSize, opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight,
           opt.winSrcIntWeight, opt.winSrcAxisWeight, opt.winSrcLambWeight )

opt.experiment = osp.join(curDir, opt.experiment )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp %s/*.py %s' % (curDir, opt.experiment ) )

shgW = opt.shadingWeight
geoW = opt.geometryWeight
sizeW = opt.sizeWeight
renW = opt.renderWeight
srcIntW = opt.winSrcIntWeight
srcAxisW = opt.winSrcAxisWeight
srcLambW = opt.winSrcLambWeight

maxWinNum = 3

opt.seed = int(opt.iterId + 16 )
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )

# Network for lighting prediction
visWinNet = modelLight.windowNet(isInv = False )

if opt.isFineTune:
    visWinDict = torch.load('{0}/visWinNet_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
    visWinNet.load_state_dict(visWinDict['model'] )

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

visWinNet = nn.DataParallel(visWinNet, device_ids = opt.deviceIds )

renderWindow = renderWindow.renderDirecLighting(sampleNum = 100 )

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
normalDecoder = normalDecoder.cuda()
visWinNet = visWinNet.cuda()

# Optimizer
opVisWin = optim.Adam(visWinNet.parameters(), lr=1e-4, betas=(0.9, 0.999 ) )
if opt.isFineTune:
    opVisWin.load_state_dict(visWinDict['optim'] )


brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isLightSrc = True, phase = 'TRAIN')
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 0, shuffle = True, drop_last = False )

j = opt.iterId

visWinPointsErrsNpList = np.ones([1, 1], dtype = np.float32 )
visWinNormalErrsNpList = np.ones([1, 1], dtype=np.float32 )
visWinSizeErrsNpList = np.ones([1, 1], dtype=np.float32 )

visWinSrcErrsNpList = np.ones([1, 3], dtype = np.float32 )
visWinSrcSkyErrsNpList = np.ones([1, 3], dtype = np.float32 )
visWinSrcGrdErrsNpList = np.ones([1, 3], dtype = np.float32 )

visWinShgErrsNpList = np.ones( [1, 1], dtype = np.float32 )
visWinShgAppErrsNpList = np.ones( [1, 2], dtype = np.float32 )
visWinRenErrsNpList = np.ones( [1, 1], dtype = np.float32 )
visWinNumNpList = np.ones([1, 1], dtype = np.float32 )

nrow = opt.batchSize * 3

epochId = int(opt.iterId / np.ceil(brdfDataset.count / float(opt.batchSize ) ) )
for epoch in list(range(epochId, opt.nepoch ) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w')
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

        # Load visible window
        visWinNum, winMasksBatch, onWinMasksBatch, \
                visWinPlanesBatch, visWinSrcsBatch, \
                visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
                visWinShadowsBatch \
                = dataLoader.getVisWindow(dataBatch )

        nameBatch = dataBatch['name']
        batchSize = imBatch.size(0 )
        for m in range(0, int(batchSize / 3.0 ) ):
            print('%s %d' % (nameBatch[m], visWinNum[m] ) )
            trainingLog.write('%s %d\n' % (nameBatch[m], visWinNum[m] ) )

        # Clear the gradient in Optimizer
        opVisWin.zero_grad()

        depthMax = torch.max(torch.max(depthBatch, dim=2, keepdim=True )[0], dim=3, keepdim=True )[0]
        depthBatch = depthBatch * segAllBatch + (1 - segAllBatch ) * depthMax
        inputBatch = torch.cat([imBatch, depthBatch], dim=1 )

        # Predict the large BRDF
        x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
        albedoPred, _ = albedoDecoder(x1, x2, x3,
                x4, x5, x6, [opt.imHeight, opt.imWidth ] )
        normalPred, _ = normalDecoder(x1, x2, x3,
                x4, x5, x6, [opt.imHeight, opt.imWidth] )

        # Down sample the image and masks
        winMasksSmallBatch = F.adaptive_avg_pool2d(winMasksBatch, (opt.envRow, opt.envCol ) )
        onWinMasksSmallBatch = F.adaptive_avg_pool2d(onWinMasksBatch, (opt.envRow, opt.envCol ) )

        segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )
        imSmallBatch = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol ) )

        albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
        normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
        depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

        # Predict the direct lighting of visible window
        visWinShadingNoPreds = []
        visWinRenderGts = []
        visWinRenderPreds = []

        visWinCenterPreds = []
        visWinNormalPreds = []
        visWinYAxisPreds = []
        visWinXAxisPreds = []

        visWinSrcPreds = []
        visWinSrcSkyPreds = []
        visWinSrcGrdPreds = []

        visWinShgErr = 0
        visWinShgAppErr = 0
        visWinShgAppOrigErr = 0
        visWinRenErr = 0

        visWinPointsErr = 0
        visWinNormalErr = 0
        visWinSizeErr = 0

        visWinSrcErr = [0, 0, 0]
        visWinSrcSkyErr = [0, 0, 0]
        visWinSrcGrdErr = [0, 0, 0]

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
                                visWinSrcSunPred, visWinSrcSkyPred, visWinSrcGrdPred \
                                = visWinNet(
                                    imSmallBatch[m:m+1, :],
                                    depthDS.detach()[m:m+1, :],
                                    albedoDS.detach()[m:m+1, :],
                                    lightOnMasksSmallBatch[m:m+1, :],
                                    onWinMasksSmallBatch[m:m+1, n:n+1]
                                )

                        visWinSrcPreds.append(visWinSrcSunPred )
                        visWinSrcSkyPreds.append(visWinSrcSkyPred )
                        visWinSrcGrdPreds.append(visWinSrcGrdPred )

                        visWinShadingNoPred \
                                = renderWindow.forward(
                                visWinCenterPred,
                                visWinNormalPred,
                                visWinYAxisPred,
                                visWinXAxisPred,
                                visWinSrcSunPred,
                                visWinSrcSkyPred,
                                visWinSrcGrdPred,
                                depthDS.detach()[m:m+1, :],
                                normalDS.detach()[m:m+1, :] )

                        visWinCenterPreds.append(visWinCenterPred )
                        visWinNormalPreds.append(visWinNormalPred )
                        visWinYAxisPreds.append(visWinYAxisPred )
                        visWinXAxisPreds.append(visWinXAxisPred )
                        visWinShadingNoPreds.append(visWinShadingNoPred )

                        visWinRenderGt = visWinShadingsNoBatch[m:m+1, n] * albedoSmallBatch[m:m+1, :]
                        visWinRenderPred = visWinShadingNoPred * albedoDS[m:m+1, :]

                        visWinRenderGts.append(visWinRenderGt )
                        visWinRenderPreds.append(visWinRenderPred )

                        visWinShgErr += torch.mean(
                            torch.abs(visWinShadingNoPred - visWinShadingsNoBatch[m: m+1, n, :] ) \
                            * segEnvBatch[m:m+1, :] * semLabelSmallBatch[m:m+1, :] )
                        visWinShgAppErr += torch.mean(
                            torch.abs(visWinShadingsNoAppBatch[m:m+1, n, :] - visWinShadingsNoBatch[m: m+1, n, :] ) \
                            * segEnvBatch[m:m+1, :] * semLabelSmallBatch[m:m+1, :] )
                        visWinShgAppOrigErr += visWinSrcsBatch[m:m+1, n, 21]

                        visWinRenErr += torch.mean(
                            torch.abs(visWinRenderPred - visWinRenderGt ) \
                            * segEnvBatch[m:m+1, :] * semLabelSmallBatch[m:m+1, :] )
                    else:
                        visWinCenterPreds.append(None )
                        visWinNormalPreds.append(None )
                        visWinYAxisPreds.append(None )
                        visWinXAxisPreds.append(None )

                        visWinSrcPreds.append(None )
                        visWinSrcSkyPreds.append(None )
                        visWinSrcGrdPreds.append(None )

                        visWinShadingNoPreds.append(
                                torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                        visWinRenderPreds.append(
                                torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                        visWinRenderGts.append(
                                torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )

            visWinPointsErr, visWinNormalErr, visWinSizeErr, visWinSrcErr, visWinSrcSkyErr, visWinSrcGrdErr \
                = lossFunctions.visWindowLoss(
                    maxWinNum,
                    visWinCenterPreds, visWinNormalPreds,
                    visWinXAxisPreds, visWinYAxisPreds,
                    visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
                    visWinOns,
                    visWinPlanesBatch, visWinSrcsBatch )

            visWinErr = geoW * (visWinPointsErr + visWinNormalErr) + sizeW * visWinSizeErr + shgW * visWinShgErr + renW * visWinRenErr
            visWinErr += (visWinSrcErr[0] * srcIntW + visWinSrcErr[1] * srcAxisW + visWinSrcErr[2] * srcLambW +
                          0.2 * (visWinSrcSkyErr[0] * srcIntW + visWinSrcSkyErr[1] * srcAxisW + visWinSrcSkyErr[2] * srcLambW ) +
                          0.2 * (visWinSrcGrdErr[0] * srcIntW + visWinSrcGrdErr[1] * srcAxisW  + visWinSrcGrdErr[2] * srcLambW ) )
            visWinErr /=  onVisWinNum
            visWinErr.backward()
            opVisWin.step()

        visWinSrcErrMean, visWinSrcSkyErrMean, visWinSrcGrdErrMean = [], [], []
        for n in range(0, 3):
            visWinSrcErrMean.append(visWinSrcErr[n] / max(onVisWinNum, 1) )
            visWinSrcSkyErrMean.append(visWinSrcSkyErr[n] / max(onVisWinNum, 1) )
            visWinSrcGrdErrMean.append(visWinSrcGrdErr[n] / max(onVisWinNum, 1) )

        # Write errors to screen
        utils.writeErrToScreen('visWinPoints', [visWinPointsErr / max(onVisWinNum, 1) ], epoch, j )
        utils.writeErrToScreen('visWinNormal', [visWinNormalErr / max(onVisWinNum, 1) ], epoch, j )
        utils.writeErrToScreen('visWinSize', [visWinSizeErr / max(onVisWinNum, 1) ], epoch, j )

        utils.writeErrToScreen('visWinSrc', visWinSrcErrMean, epoch, j )
        utils.writeErrToScreen('visWinSrcSky', visWinSrcSkyErrMean, epoch, j )
        utils.writeErrToScreen('visWinSrcGrd', visWinSrcGrdErrMean, epoch, j )

        utils.writeErrToScreen('visWinShading', [visWinShgErr / max(onVisWinNum, 1) ], epoch, j )
        utils.writeErrToScreen('visWinShadingApp', [visWinShgAppErr / max(onVisWinNum, 1), visWinShgAppOrigErr / max(onVisWinNum, 1) ], epoch, j )
        utils.writeErrToScreen('visWinRender', [visWinRenErr / max(onVisWinNum, 1) ], epoch, j )

        # Write errors to file
        utils.writeErrToFile('visWinPoints', [visWinPointsErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )
        utils.writeErrToFile('visWinNormal', [visWinNormalErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )
        utils.writeErrToFile('visWinSize', [visWinSizeErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )

        utils.writeErrToFile('visWinSrc', visWinSrcErrMean, trainingLog, epoch, j )
        utils.writeErrToFile('visWinSrcSky', visWinSrcSkyErrMean, trainingLog, epoch, j )
        utils.writeErrToFile('visWinSrcGrd', visWinSrcGrdErrMean, trainingLog, epoch, j )

        utils.writeErrToFile('visWinShading', [visWinShgErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )
        utils.writeErrToFile('visWinShadingApp', [visWinShgAppErr / max(onVisWinNum, 1), visWinShgAppOrigErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )
        utils.writeErrToFile('visWinRender', [visWinRenErr / max(onVisWinNum, 1) ], trainingLog, epoch, j )

        # Accumulate errors
        visWinPointsErrsNpList = np.concatenate( [visWinPointsErrsNpList, utils.turnErrorIntoNumpy( [visWinPointsErr] )], axis=0 )
        visWinNormalErrsNpList = np.concatenate( [visWinNormalErrsNpList, utils.turnErrorIntoNumpy( [visWinNormalErr] )], axis=0 )
        visWinSizeErrsNpList = np.concatenate( [visWinSizeErrsNpList, utils.turnErrorIntoNumpy( [visWinSizeErr] )], axis=0 )

        visWinSrcErrsNpList = np.concatenate( [visWinSrcErrsNpList, utils.turnErrorIntoNumpy( visWinSrcErr )], axis=0 )
        visWinSrcSkyErrsNpList = np.concatenate( [visWinSrcSkyErrsNpList, utils.turnErrorIntoNumpy( visWinSrcSkyErr )], axis=0 )
        visWinSrcGrdErrsNpList = np.concatenate( [visWinSrcGrdErrsNpList, utils.turnErrorIntoNumpy( visWinSrcGrdErr )], axis=0 )

        visWinShgErrsNpList = np.concatenate( [visWinShgErrsNpList, utils.turnErrorIntoNumpy( [visWinShgErr] ) ], axis=0 )
        visWinShgAppErrsNpList = np.concatenate( [visWinShgAppErrsNpList, utils.turnErrorIntoNumpy( [visWinShgAppErr, visWinShgAppOrigErr ] ) ], axis=0 )
        visWinRenErrsNpList = np.concatenate( [visWinRenErrsNpList, utils.turnErrorIntoNumpy( [visWinRenErr] ) ], axis=0 )
        visWinNumNpList = np.concatenate( [visWinNumNpList, utils.turnErrorIntoNumpy( [onVisWinNum ] ) ], axis=0 )

        torch.cuda.empty_cache()

        if j - opt.iterId < 5000:
            # Write errors to screen
            visWinNumSum = max(np.sum(visWinNumNpList[1: j+1-opt.iterId ] ), 1 )

            utils.writeNpErrToScreen('visWinPointsAccu', np.sum(visWinPointsErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinNormalAccu',np.sum(visWinNormalErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSizeAccu',np.sum(visWinSizeErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            utils.writeNpErrToScreen('visWinSrcAccu', np.sum(visWinSrcErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSrcSkyAccu', np.sum(visWinSrcSkyErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSrcGrdAccu', np.sum(visWinSrcGrdErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            utils.writeNpErrToScreen('visWinShgAccu', np.sum(visWinShgErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinShgAppAccu', np.sum(visWinShgAppErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinRenAccu', np.sum(visWinRenErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            # Write errors to file
            utils.writeNpErrToFile('visWinPointsAccu', np.sum(visWinPointsErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinNormalAccu', np.sum(visWinNormalErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSizeAccu', np.sum(visWinSizeErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )

            utils.writeNpErrToFile('visWinSrcAccu', np.sum(visWinSrcErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSrcSkyAccu', np.sum(visWinSrcSkyErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSrcGrdAccu', np.sum(visWinSrcGrdErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )

            utils.writeNpErrToFile('visWinShgAccu', np.sum(visWinShgErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinShgAppAccu', np.sum(visWinShgAppErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinRenAccu', np.sum(visWinRenErrsNpList[1:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
        else:
            visWinNumSum = max(np.sum(visWinNumNpList[j-4999-opt.iterId: j+1-opt.iterId ] ), 1 )

            utils.writeNpErrToScreen('visWinPointsAccu', np.sum(visWinPointsErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinNormalAccu', np.sum(visWinNormalErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSizeAccu', np.sum(visWinSizeErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            utils.writeNpErrToScreen('visWinSrcAccu', np.sum(visWinSrcErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSrcSkyAccu', np.sum(visWinSrcSkyErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinSrcGrdAccu', np.sum(visWinSrcGrdErrsNpList[j-4999-opt.iterId: j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            utils.writeNpErrToScreen('visWinShgAccu', np.sum(visWinShgErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinShgAppAccu', np.sum(visWinShgAppErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )
            utils.writeNpErrToScreen('visWinRenAccu', np.sum(visWinRenErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, epoch, j )

            # Write errors to file
            utils.writeNpErrToFile('visWinPointsAccu', np.sum(visWinPointsErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinNormalAccu', np.sum(visWinNormalErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSizeAccu', np.sum(visWinSizeErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )

            utils.writeNpErrToFile('visWinSrcAccu', np.sum(visWinSrcErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSrcSkyAccu', np.sum(visWinSrcSkyErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinSrcGrdAccu', np.sum(visWinSrcGrdErrsNpList[j-4999-opt.iterId: j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )

            utils.writeNpErrToFile('visWinShgAccu', np.sum(visWinShgErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinShgAppAccu', np.sum(visWinShgAppErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )
            utils.writeNpErrToFile('visWinRenAccu', np.sum(visWinRenErrsNpList[j-4999-opt.iterId:j+1-opt.iterId, :], axis=0) / visWinNumSum, trainingLog, epoch, j )

        if j == 1 or j% 2000 == 0:
            # Save the groundtruth results
            vutils.save_image( albedoBatch ** (1.0/2.2), '{0}/{1}_albedoGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( 0.5*(normalBatch + 1), '{0}/{1}_normalGt.png'.format(opt.experiment, j), nrow=nrow )
            depthOutGt = 1 / torch.clamp(depthBatch + 1, 1e-6, 10 )
            vutils.save_image( depthOutGt, '{0}/{1}_depthGt.png'.format(opt.experiment, j), nrow=nrow )

            # Save the predicted BRDF
            vutils.save_image( albedoPred ** (1.0/2.2), '{0}/{1}_albedoPred.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( 0.5*(normalPred + 1), '{0}/{1}_normalPred.png'.format(opt.experiment, j), nrow=nrow )

            # Output Ground-truth point clouds
            normalSmallBatch = F.adaptive_avg_pool2d(normalBatch, [opt.envRow, opt.envCol ] )
            utils.writeDepthAsPointClouds(
                    depthDS,
                    normalSmallBatch,
                    segEnvBatch,
                    '{0}/{1}_roomGt.ply'.format(opt.experiment, j) )

            # Output the predicted point clouds
            utils.writeDepthAsPointClouds(
                    depthDS,
                    normalDS,
                    segEnvBatch,
                    '{0}/{1}_roomPred.ply'.format(opt.experiment, j) )

            # Output Ground-truth window
            if onVisWinNum > 0:
                utils.writeWindowBatch(
                        visWinPlanesBatch[:, :, 0:3],
                        visWinPlanesBatch[:, :, 6:9],
                        visWinPlanesBatch[:, :, 9:12],
                        visWinOns,
                        maxWinNum,
                        '{0}/{1}_visWinGt.obj'.format(opt.experiment, j) )
                utils.writeWindowList(
                        visWinCenterPreds,
                        visWinYAxisPreds,
                        visWinXAxisPreds,
                        maxWinNum,
                        '{0}/{1}_visWinPred.obj'.format(opt.experiment, j ) )

                vutils.save_image( (visWinShadingsNoBatch ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                    '{0}/{1}_visWinShadingNoGt.png'.format(opt.experiment, j), nrow=nrow )
                vutils.save_image( (visWinShadingsNoAppBatch ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                    '{0}/{1}_visWinShadingNoApp.png'.format(opt.experiment, j), nrow=nrow )
                visWinShadingNoPreds = torch.cat(visWinShadingNoPreds, dim=0 ).reshape( batchSize, maxWinNum, 3, opt.envRow, opt.envCol )
                vutils.save_image( (visWinShadingNoPreds ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                    '{0}/{1}_visWinShadingNoPred.png'.format(opt.experiment, j), nrow=nrow )

                visWinRenderGts = torch.cat(visWinRenderGts, dim=0 ).reshape(batchSize, maxWinNum, 3, opt.envRow, opt.envCol )
                visWinRenderPreds = torch.cat(visWinRenderPreds, dim=0 ).reshape( batchSize, maxWinNum, 3, opt.envRow, opt.envCol )
                vutils.save_image( (visWinRenderGts ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                    '{0}/{1}_visWinRenderGt.png'.format(opt.experiment, j), nrow=nrow )
                vutils.save_image( (visWinRenderPreds ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxWinNum, 3, opt.envRow, opt.envCol ),
                    '{0}/{1}_visWinRenderPred.png'.format(opt.experiment, j), nrow=nrow )

                vutils.save_image( winMasksBatch.transpose(0, 1).reshape(batchSize * maxWinNum, 1, opt.imHeight, opt.imWidth ),
                    '{0}/{1}_visWinMask.png'.format(opt.experiment, j), nrow=nrow )
                vutils.save_image( onWinMasksBatch.transpose(0, 1).reshape(batchSize * maxWinNum, 1, opt.imHeight, opt.imWidth ),
                    '{0}/{1}_visOnWinMask.png'.format(opt.experiment, j), nrow=nrow )

            # Output the groundtruth lighting and image
            vutils.save_image( (imBatch )**(1.0/2.2), '{0}/{1}_im.png'.format(opt.experiment, j ), nrow=nrow )

            vutils.save_image( semLabelBatch, '{0}/{1}_semLabel.png'.format(opt.experiment, j ), nrow=nrow )

        if j  % 5000 == 0:
            # save the models
            torch.save( {'model': visWinNet.module.state_dict(), 'optim': opVisWin.state_dict() },
                       '{0}/visWinNet_iter{1}.pth'.format(opt.experiment, j ) )

    trainingLog.close()

    # Save the error record
    np.save('{0}/visWinPointsError_{1}.npy'.format(opt.experiment, epoch), visWinPointsErrsNpList )
    np.save('{0}/visWinNormalError_{1}.npy'.format(opt.experiment, epoch), visWinNormalErrsNpList )
    np.save('{0}/visWinSizeError_{1}.npy'.format(opt.experiment, epoch), visWinSizeErrsNpList )

    np.save('{0}/visWinSrcError_{1}.npy'.format(opt.experiment, epoch), visWinSrcErrsNpList )
    np.save('{0}/visWinSrcSkyError_{1}.npy'.format(opt.experiment, epoch), visWinSrcSkyErrsNpList )
    np.save('{0}/visWinSrcGrdError_{1}.npy'.format(opt.experiment, epoch), visWinSrcGrdErrsNpList )

    np.save('{0}/visWinShgError_{1}.npy'.format(opt.experiment, epoch), visWinShgErrsNpList )
    np.save('{0}/visWinShgAppError_{1}.npy'.format(opt.experiment, epoch), visWinShgAppErrsNpList )
    np.save('{0}/visWinRenError_{1}.npy'.format(opt.experiment, epoch), visWinRenErrsNpList )
    np.save('{0}/visWinNum_{1}.npy'.format(opt.experiment, epoch), visWinNumNpList )

torch.save( {'model': visWinNet.module.state_dict(), 'optim': opVisWin.state_dict() },
           '{0}/visWinNet_iter{1}.pth'.format(opt.experiment, j ) )
