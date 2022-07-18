import utils
import torch
import numpy as np
import argparse
import random
import os
import models
import modelLight
import renderVisLamp
import torchvision.utils as vutils
import dataLoaderVisLamp as dataLoader
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
parser.add_argument('--testRoot', default=None, help='the path to store testing results')
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction')
parser.add_argument('--batchSize', type=int, default=3, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the height / width of the input image to network' )
parser.add_argument('--envCol', type=int, default=160, help='the height / width of the input image to network' )
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network' )

# Finetuning parameters
parser.add_argument('--iterId', type=int, default=119540, help='the iteration used for fine-tuning' )

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )

parser.add_argument('--isPredDepth', action='store_true', help='whether to use predicted depth or not')

# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_visLamp'
    opt.experiment += '_bn%d_shg%.3f_geo%.3f_ren%.3f' \
        % (opt.batchSize, opt.shadingWeight, opt.geometryWeight, opt.renderWeight )

opt.experiment = osp.join(curDir, opt.experiment )
if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check_', 'test_')
    if opt.isPredDepth:
        opt.testRoot += '_predDepth'
os.system('mkdir {0}'.format(opt.testRoot) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )

maxLampNum = 7

opt.seed = 32
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

# Network for lighting prediction
visLampNet = modelLight.lampNet(isInv = False )
visLampDict = torch.load('{0}/visLampNet_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
visLampNet.load_state_dict(visLampDict['model'] )
for param in visLampNet.parameters():
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
visLampNet = nn.DataParallel(visLampNet, device_ids = opt.deviceIds )

renderVisLamp = renderVisLamp.renderDirecLighting()

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
roughDecoder = roughDecoder.cuda()
normalDecoder = normalDecoder.cuda()
visLampNet = visLampNet.cuda()

brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isLightSrc = True, phase = 'TEST', isPredDepth = opt.isPredDepth )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 0, shuffle = False, drop_last = False )

j = 0

visLampPointsErrsNpList = np.ones([1, 1], dtype = np.float32 )
visLampShgErrsNpList = np.ones( [1, 1], dtype = np.float32 )
visLampRenErrsNpList = np.ones( [1, 1], dtype = np.float32 )
visLampNumNpList = np.ones([1, 1], dtype = np.float32 )

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

    # Load visible lamp
    visLampNum, lampMasksBatch, onLampMasksBatch, \
            visLampCentersBatch, visLampAxesBatch, \
            visLampShadingsNoBatch, \
            visLampShadowsBatch \
            = dataLoader.getVisLamp(dataBatch )

    nameBatch = dataBatch['name']
    batchSize = imBatch.size(0 )
    for m in range(0, int(batchSize / 3.0 ) ):
        print('%s %d' % (nameBatch[m], visLampNum[m] ) )
        testingLog.write('%s %d\n' % (nameBatch[m], visLampNum[m] ) )

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
    lampMasksSmallBatch = F.adaptive_avg_pool2d(lampMasksBatch, (opt.envRow, opt.envCol ) )
    onLampMasksSmallBatch = F.adaptive_avg_pool2d(onLampMasksBatch, (opt.envRow, opt.envCol ) )

    segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )
    imSmallBatch = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol ) )

    albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
    normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
    depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

    # Predict the direct lighting of visible lamp
    visLampShadingNoPreds = []
    visLampRenderGts = []
    visLampRenderPreds = []

    visLampCenterPreds = []
    visLampPointsPreds = []

    visLampShgErr = 0
    visLampRenErr = 0
    visLampPointsErr = 0
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
                    visLampShadingNoPred, visLampPointsPred \
                            = renderVisLamp.forward(
                            visLampCenterPred,
                            visLampSrcPred,
                            depthDS.detach()[m:m+1, :],
                            onLampMasksSmallBatch[m:m+1, n:n+1, :],
                            normalDS.detach()[m:m+1, :],
                            isTest = False )

                    visLampCenterPreds.append(visLampCenterPred )
                    visLampPointsPreds.append(visLampPointsPred[0] )
                    visLampShadingNoPreds.append(visLampShadingNoPred )

                    visLampRenderGt = visLampShadingsNoBatch[m:m+1, n] * albedoSmallBatch[m:m+1, :]
                    visLampRenderPred = visLampShadingNoPred * albedoDS[m:m+1, :]

                    visLampRenderGts.append(visLampRenderGt )
                    visLampRenderPreds.append(visLampRenderPred )

                    visLampShgErr += torch.mean(
                        torch.abs(visLampShadingNoPred - visLampShadingsNoBatch[m:m+1, n] ) \
                        * segEnvBatch[m:m+1, :] )
                    visLampRenErr += torch.mean(
                        torch.abs(visLampRenderPred - visLampRenderGt ) \
                        * segEnvBatch[m:m+1, :] )
                else:
                    visLampCenterPreds.append(None )
                    visLampPointsPreds.append(None )
                    visLampShadingNoPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                    visLampRenderPreds.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )
                    visLampRenderGts.append(
                            torch.zeros([1, 3, opt.envRow, opt.envCol ], dtype = torch.float32 ).cuda() )

        visLampPointsErr = lossFunctions.visLampLoss(
                maxLampNum,
                visLampPointsPreds,
                visLampOns,
                visLampAxesBatch, visLampCentersBatch,
                depthScaleBatch,
                isTest = True
        )

    # Write errors to screen
    utils.writeErrToScreen('visLampPoints', [visLampPointsErr / max(onVisLampNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('visLampShading', [visLampShgErr / max(onVisLampNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('visLampRender', [visLampRenErr / max(onVisLampNum, 1) ], opt.iterId, j )

    # Write errors to file
    utils.writeErrToFile('visLampPoints', [visLampPointsErr / max(onVisLampNum, 1) ], testingLog, opt.iterId, j )
    utils.writeErrToFile('visLampShading', [visLampShgErr / max(onVisLampNum, 1) ], testingLog , opt.iterId, j )
    utils.writeErrToFile('visLampRender', [visLampRenErr / max(onVisLampNum, 1) ], testingLog , opt.iterId, j )

    # Accumulate errors
    visLampPointsErrsNpList = np.concatenate( [visLampPointsErrsNpList, utils.turnErrorIntoNumpy( [visLampPointsErr] )], axis=0 )
    visLampShgErrsNpList = np.concatenate( [visLampShgErrsNpList, utils.turnErrorIntoNumpy( [visLampShgErr] ) ], axis=0 )
    visLampRenErrsNpList = np.concatenate( [visLampRenErrsNpList, utils.turnErrorIntoNumpy( [visLampRenErr] ) ], axis=0 )
    visLampNumNpList = np.concatenate( [visLampNumNpList, utils.turnErrorIntoNumpy( [onVisLampNum ] ) ], axis=0 )

    torch.cuda.empty_cache()

    visLampNumSum = max(np.sum(visLampNumNpList[1: j+1 ] ), 1 )

    utils.writeNpErrToScreen('visLampPointsAccu', np.sum(visLampPointsErrsNpList[1:j+1, :], axis=0) / visLampNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('visLampShgAccu', np.sum(visLampShgErrsNpList[1:j+1, :], axis=0) / visLampNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('visLampRenAccu', np.sum(visLampRenErrsNpList[1:j+1, :], axis=0) / visLampNumSum, opt.iterId, j )

    # Write errors to file
    utils.writeNpErrToFile('visLampPointsAccu', np.sum(visLampPointsErrsNpList[1:j+1, :], axis=0) / visLampNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('visLampShgAccu', np.sum(visLampShgErrsNpList[1:j+1, :], axis=0) / visLampNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('visLampRenAccu', np.sum(visLampRenErrsNpList[1:j+1, :], axis=0) / visLampNumSum, testingLog, opt.iterId, j )

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

        # Output Ground-truth lamp
        if onVisLampNum > 0:
            utils.writeLampBatch(
                    visLampAxesBatch,
                    visLampCentersBatch,
                    visLampOns,
                    maxLampNum,
                    '{0}/{1}_visLampGt.ply'.format(opt.testRoot, j) )
            utils.writeLampList(
                    visLampCenterPreds,
                    depthDS,
                    normalDS,
                    onLampMasksSmallBatch,
                    maxLampNum,
                    '{0}/{1}_visLampPred.ply'.format(opt.testRoot, j) )

            vutils.save_image( (visLampShadingsNoBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadingNoGt.png'.format(opt.testRoot, j), nrow=nrow )

            visLampShadingNoPreds = torch.cat(visLampShadingNoPreds, dim=0 ).reshape(batchSize, maxLampNum, 3, opt.envRow, opt.envCol )
            vutils.save_image( (visLampShadingNoPreds ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

            visLampRenderGts = torch.cat(visLampRenderGts, dim=0 ).reshape( batchSize, maxLampNum, 3, opt.envRow, opt.envCol )
            visLampRenderPreds = torch.cat(visLampRenderPreds, dim=0 ).reshape( batchSize, maxLampNum, 3, opt.envRow, opt.envCol )
            vutils.save_image( (visLampRenderGts ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampRenderGt.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( (visLampRenderPreds ** (1.0/2.2) ).transpose(0, 1).reshape( batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
                '{0}/{1}_visLampRenderPred.png'.format(opt.testRoot, j), nrow=nrow )

            vutils.save_image( lampMasksBatch.transpose(0, 1).reshape(batchSize * maxLampNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visLampMask.png'.format(opt.testRoot, j), nrow=nrow )
            vutils.save_image( onLampMasksBatch.transpose(0, 1).reshape(batchSize * maxLampNum, 1, opt.imHeight, opt.imWidth ),
                '{0}/{1}_visOnLampMask.png'.format(opt.testRoot, j), nrow=nrow )

        # Output the groundtruth lighting and image
        vutils.save_image( (imBatch )**(1.0/2.2), '{0}/{1}_im.png'.format(opt.testRoot, j ), nrow=nrow )

testingLog.close()

# Save the error record
np.save('{0}/visLampPointsError_iter{1}.npy'.format(opt.testRoot, opt.iterId), visLampPointsErrsNpList )
np.save('{0}/visLampShgError_iter{1}.npy'.format(opt.testRoot, opt.iterId), visLampShgErrsNpList )
np.save('{0}/visLampRenError_iter{1}.npy'.format(opt.testRoot, opt.iterId), visLampRenErrsNpList )
np.save('{0}/visLampNum_iter{1}.npy'.format(opt.testRoot, opt.iterId), visLampNumNpList )
