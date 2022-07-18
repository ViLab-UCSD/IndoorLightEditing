import utils
import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import models
import modelLight
import renderInvLamp
import torchvision.utils as vutils
import torchvision.models as vmodels
import dataLoaderInvLamp as dataLoader
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
parser.add_argument('--testRoot', default=None, help='the path to store testing results' )
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
parser.add_argument('--iterId', type=int, default=150000, help='the iteration used for fine-tuning' )

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1.0, help='the weight for shading error' )
parser.add_argument('--geometryWeight', type=float, default=1.0, help='the weight for geometry error' )
parser.add_argument('--renderWeight', type=float, default=1.0, help='the weight for render error' )
parser.add_argument('--sizeWeight', type=float, default=0.2, help='the weight for the size error')

parser.add_argument('--isPredDepth', action='store_true', help='whether to use predicted depth or not')

# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_invLamp'
    opt.experiment += '_bn%d_shg%.3f_geo%.3f_size%.3f_ren%.3f' \
        % (opt.batchSize, opt.shadingWeight, opt.geometryWeight, opt.sizeWeight, opt.renderWeight )
opt.experiment = osp.join(curDir, opt.experiment )
if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check_', 'test_')
    if opt.isPredDepth:
        opt.testRoot += '_predDepth'
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )

maxLampNum = 7

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
invLampNet = modelLight.lampNet(isInv = True )
invLampDict = torch.load('{0}/invLampNet_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
invLampNet.load_state_dict(invLampDict['model'] )
for param in invLampNet.parameters():
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

invLampNet = nn.DataParallel(invLampNet, device_ids = opt.deviceIds )

renderInvLamp = renderInvLamp.renderDirecLighting()

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
roughDecoder = roughDecoder.cuda()
normalDecoder = normalDecoder.cuda()

invLampNet = invLampNet.cuda()

brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isLightSrc = True, phase = 'TEST', isPredDepth = opt.isPredDepth )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 0, shuffle = False, drop_last = False )

j = 0
invLampPointsErrsNpList = np.ones([1, 1], dtype = np.float32 )
invLampSizeErrsNpList = np.ones([1, 1], dtype = np.float32 )
invLampShgErrsNpList = np.ones( [1, 1], dtype = np.float32 )
invLampRenErrsNpList = np.ones( [1, 1], dtype = np.float32 )
invLampNumNpList = np.ones([1, 1], dtype = np.float32 )

nrow = opt.batchSize * 3

testingLog = open('{0}/testingLog_iter{1}.txt'.format(opt.testRoot, opt.iterId), 'w')
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

    # Load invisible lamp
    invLampNum, invLampOnBatch, \
            invLampCentersBatch, invLampAxesBatch, \
            invLampShadingsBatch, \
            invLampShadingsNoBatch, invLampShadowsBatch \
            = dataLoader.getInvLamp(dataBatch )

    nameBatch = dataBatch['name']
    batchSize = imBatch.size(0 )
    for m in range(0, int(batchSize / 3.0 ) ):
        print('%s %d' % (nameBatch[m], invLampNum[m] ) )
        testingLog.write('%s %d\n' % (nameBatch[m], invLampNum[m] ) )

    onInvLampNum = 0

    invLampCentersGt, invLampAxesGt, invLampOnGt = [], [], []
    invLampShadingGt, invLampShadingNoGt, invLampShadowGt = [], [], []
    for m in range(0, batchSize ):
        maxInt = -1
        lampId = 0
        for n in range(0, maxLampNum ):
            if invLampOnBatch[m, n] > 0:
                lampInt = torch.mean(invLampShadingsBatch[m, n] )
                lampInt = lampInt.detach().cpu().item()
                if lampInt > maxInt:
                    maxInt = lampInt
                    lampId = n

        if maxInt < 0.01:
            invLampOnGt.append(0 )
            invLampShadingGt.append(torch.zeros(1, 3, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
            invLampShadingNoGt.append(torch.zeros(1, 3, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
            invLampShadowGt.append(torch.ones(1, 1, opt.envRow, opt.envCol, dtype = torch.float32 ).cuda() )
        else:
            onInvLampNum += 1
            invLampOnGt.append(1 )
            invLampShadingGt.append(invLampShadingsNoBatch[m:m+1, lampId, :] * invLampShadowsBatch[m:m+1, lampId, :] )
            invLampShadingNoGt.append(invLampShadingsNoBatch[m:m+1, lampId, :] )
            invLampShadowGt.append(invLampShadowsBatch[m:m+1, lampId, :] )

        invLampCentersGt.append(invLampCentersBatch[m:m+1, lampId ] )
        invLampAxesGt.append(invLampAxesBatch[m:m+1, lampId ] )


    invLampCentersGt = torch.cat(invLampCentersGt, dim=0 )
    invLampAxesGt = torch.cat(invLampAxesGt, dim=0 )

    invLampShadingGt = torch.cat(invLampShadingGt, dim=0 )
    invLampShadingNoGt = torch.cat(invLampShadingNoGt, dim=0 )
    invLampShadowGt = torch.cat(invLampShadowGt, dim=0 )

    # Clear the gradient in Optimizer
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

    albedoDS = albedoDS.detach()
    normalDS = normalDS.detach()

    # Predict the direct lighting of invisible lamp
    invLampAxesPred, invLampCenterPred, invLampSrcPred \
            = invLampNet(
                    imSmallBatch,
                    depthDS.detach(), albedoDS.detach(),
                    lightOnMasksSmallBatch
            )
    invLampShadingNoPred = renderInvLamp.forward(
                    invLampAxesPred,
                    invLampCenterPred,
                    invLampSrcPred,
                    depthDS.detach(),
                    normalDS.detach(),
                    isTest = False )

    # Compute losses for invisible light sources
    invLampPointsErr, invLampSizeErr = lossFunctions.invLampLoss(
                    invLampAxesPred, invLampCenterPred,
                    invLampOnGt,
                    invLampAxesGt, invLampCentersGt,
                    depthScaleBatch, isTest = True )

    invLampShadingPred = invLampShadingNoPred * invLampShadowGt
    invLampShgErr = torch.mean( torch.abs(invLampShadingNoPred - invLampShadingNoGt ) * segEnvBatch )

    invLampRenderGt = invLampShadingNoGt * albedoSmallBatch
    invLampRenderPred = invLampShadingNoPred * albedoDS
    invLampRenErr = torch.mean( torch.abs(invLampRenderPred - invLampRenderGt ) * segEnvBatch )

    # Write errors to screen
    utils.writeErrToScreen('invLampPoints', [invLampPointsErr / max(onInvLampNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('invLampSize', [invLampSizeErr / max(onInvLampNum, 1) ], opt.iterId, j )
    utils.writeErrToScreen('invLampShading', [invLampShgErr ], opt.iterId, j )
    utils.writeErrToScreen('invLampRender', [invLampRenErr ], opt.iterId, j )

    # Write errors to file
    utils.writeErrToFile('invLampPoints', [invLampPointsErr / max(onInvLampNum, 1) ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invLampSize', [invLampSizeErr / max(onInvLampNum, 1) ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invLampShading', [invLampShgErr ], testingLog, opt.iterId, j )
    utils.writeErrToFile('invLampRender', [invLampRenErr ], testingLog, opt.iterId, j )

    # Accumulate errors
    invLampPointsErrsNpList = np.concatenate( [invLampPointsErrsNpList, utils.turnErrorIntoNumpy( [invLampPointsErr] )], axis=0 )
    invLampSizeErrsNpList = np.concatenate( [invLampSizeErrsNpList, utils.turnErrorIntoNumpy( [invLampSizeErr] )], axis=0 )
    invLampShgErrsNpList = np.concatenate( [invLampShgErrsNpList, utils.turnErrorIntoNumpy( [invLampShgErr] ) ], axis=0 )
    invLampRenErrsNpList = np.concatenate( [invLampRenErrsNpList, utils.turnErrorIntoNumpy( [invLampRenErr] ) ], axis=0 )
    invLampNumNpList = np.concatenate([invLampNumNpList, utils.turnErrorIntoNumpy([onInvLampNum ] ) ], axis=0 )

    torch.cuda.empty_cache()

    # Write errors to screen
    invLampNumSum = max(np.sum(invLampNumNpList[1 : j+1 ] ), 1 )
    utils.writeNpErrToScreen('invLampPointsAccu', np.sum(invLampPointsErrsNpList[1:j+1, :], axis=0) / invLampNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invLampSizeAccu', np.sum(invLampSizeErrsNpList[1:j+1, :], axis=0) / invLampNumSum, opt.iterId, j )
    utils.writeNpErrToScreen('invLampShgAccu', np.mean(invLampShgErrsNpList[1:j+1, :], axis=0), opt.iterId, j )
    utils.writeNpErrToScreen('invLampRenAccu', np.mean(invLampRenErrsNpList[1:j+1, :], axis=0), opt.iterId, j )

    # Write errors to file
    utils.writeNpErrToFile('invLampPointsAccu', np.sum(invLampPointsErrsNpList[1:j+1, :], axis=0) / invLampNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invLampSizeAccu', np.sum(invLampSizeErrsNpList[1:j+1, :], axis=0) / invLampNumSum, testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invLampShgAccu', np.mean(invLampShgErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )
    utils.writeNpErrToFile('invLampRenAccu', np.mean(invLampRenErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )

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
        utils.writeLampBatch(
                invLampAxesBatch,
                invLampCentersBatch,
                invLampOnBatch.detach().cpu().numpy(),
                maxLampNum,
                '{0}/{1}_invLampGt.ply'.format(opt.testRoot, j ) )
        vutils.save_image( (invLampShadingsNoBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
            '{0}/{1}_invLampShadingNoGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invLampShadingsBatch ** (1.0/2.2) ).transpose(0, 1).reshape(batchSize * maxLampNum, 3, opt.envRow, opt.envCol ),
            '{0}/{1}_invLampShadingGt.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( (invLampShadingNoGt ** (1.0/2.2) ), '{0}/{1}_invLampShadingNoGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invLampShadingNoPred ** (1.0/2.2) ), '{0}/{1}_invLampShadingNoPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( (invLampShadingGt ** (1.0/2.2) ), '{0}/{1}_invLampShadingGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invLampShadingPred ** (1.0/2.2) ), '{0}/{1}_invLampShadingPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( (invLampRenderGt ** (1.0/2.2) ), '{0}/{1}_invLampRenderGtSelected.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( (invLampRenderPred ** (1.0/2.2) ), '{0}/{1}_invLampRenderPred.png'.format(opt.testRoot, j), nrow=nrow )

        utils.writeLampBatch(
                invLampAxesPred.unsqueeze(1),
                invLampCenterPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                '{0}/{1}_invLampPred.ply'.format(opt.testRoot, j), )

        # Output the groundtruth lighting and image
        vutils.save_image( (imBatch )**(1.0/2.2), '{0}/{1}_im.png'.format(opt.testRoot, j ), nrow=nrow )

testingLog.close()

# Save the error record
np.save('{0}/invLampPointsError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invLampPointsErrsNpList )
np.save('{0}/invLampSizeError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invLampSizeErrsNpList )
np.save('{0}/invLampShgError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invLampShgErrsNpList )
np.save('{0}/invLampRenError_iter{1}.npy'.format(opt.testRoot, opt.iterId), invLampRenErrsNpList )
np.save('{0}/invLampNum_iter{1}.npy'.format(opt.testRoot, opt.iterId), invLampNumNpList )

