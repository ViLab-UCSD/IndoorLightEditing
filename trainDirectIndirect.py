import utils
import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import models
import modelLight
import torchvision.utils as vutils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images')
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=6, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')

# Finetuning parameters
parser.add_argument('--isFineTune', action='store_true', help='fine tune the network for global local lighting prediction')
parser.add_argument('--iterId', type=int, default=0, help='the iteration used for fine-tuning')

# The training weight
parser.add_argument('--shadingWeight', type=float, default=1, help='the weight for rendering error' )
parser.add_argument('--renderWeight', type=float, default=1, help='the weight for rendering error' )
# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_directIndirect'

opt.experiment = osp.join(curDir, opt.experiment )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp %s/*.py %s' % (curDir, opt.experiment ) )

shgW = opt.shadingWeight
renW = opt.renderWeight

opt.seed = 32
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

indirectLightNet = modelLight.indirectLightNet()

if opt.isFineTune:
    indirectLightDict = torch.load('{0}/indirectLightNet_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
    indirectLightNet.load_state_dict(indirectLightDict['model'] )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

encoder.load_state_dict(
    torch.load('{0}/encoder_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
albedoDecoder.load_state_dict(
    torch.load('{0}/albedo_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
normalDecoder.load_state_dict(
    torch.load('{0}/normal_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
roughDecoder.load_state_dict(
    torch.load('{0}/rough_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
for param in encoder.parameters():
    param.requires_grad = False
for param in albedoDecoder.parameters():
    param.requires_grad = False
for param in normalDecoder.parameters():
    param.requires_grad = False
for param in roughDecoder.parameters():
    param.requires_grad = False

encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )

indirectLightNet = nn.DataParallel(indirectLightNet, device_ids = opt.deviceIds )

# Send things into GPU
encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
normalDecoder = normalDecoder.cuda()
roughDecoder = roughDecoder.cuda()

indirectLightNet = indirectLightNet.cuda()

# Optimizer
opIndirectLight = optim.Adam(indirectLightNet.parameters(), lr=1e-4, betas=(0.9, 0.999 ) )
if opt.isFineTune:
    opIndirectLight.load_state_dict(indirectLightDict['optim'] )


brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, rseed = opt.seed,
        isShading = True, isLightSrc = True, phase = 'TRAIN')
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 6, shuffle = True, drop_last = False )

j = opt.iterId
shadingErrsNpList = np.ones( [1, 2], dtype = np.float32 )

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

        albedoSmallBatch = F.adaptive_avg_pool2d( albedoBatch, (opt.envRow, opt.envCol ) )

        # Load image
        im_cpu = dataBatch['im']
        imBatch = im_cpu.cuda()
        imDl_cpu = dataBatch['imDl']
        imDlBatch = imDl_cpu.cuda()
        imDm_cpu = dataBatch['imDm']
        imDmBatch = imDm_cpu.cuda()
        imBatch = torch.cat([imBatch, imDlBatch, imDmBatch], dim=0 )

        lightOnMasks_cpu = dataBatch['lightOnMasks']
        lightOnMasksBatch = lightOnMasks_cpu.cuda()
        lightOnMasksDl_cpu = dataBatch['lightOnMasksDl']
        lightOnMasksDlBatch = lightOnMasksDl_cpu.cuda()
        lightOnMasksBatch = torch.cat([lightOnMasksBatch,
            lightOnMasksDlBatch, lightOnMasksBatch ], dim=0 )

        lightOnMasksSmallBatch = F.adaptive_avg_pool2d(lightOnMasksBatch, (opt.envRow, opt.envCol ) )

        # Load shading
        shadingBatch, shadingDirectBatch \
                = dataLoader.getShading(dataBatch )

        nameBatch = dataBatch['name']
        batchSize = imBatch.size(0 )

        # Clear the gradient in Optimizer
        opIndirectLight.zero_grad()

        # Predict the large BRDF
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
        segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )

        albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
        normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
        roughDS = F.adaptive_avg_pool2d(roughPred, (opt.envRow, opt.envCol ) )
        depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

        segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )

        # Predict the global illumination
        shadingDirectBatchInput = torch.atan(shadingDirectBatch ) / np.pi * 2.0

        shadingIndirectPred = indirectLightNet(
                albedoDS.detach(),
                normalDS.detach(),
                depthDS.detach(),
                shadingDirectBatchInput.detach(),
                lightOnMasksSmallBatch )

        shadingPred = shadingIndirectPred + shadingDirectBatch.detach()

        renderedPred = shadingPred * albedoDS
        renderedGt = shadingBatch * albedoSmallBatch

        shgErrs = []
        loss1 = torch.mean( torch.abs(shadingPred - shadingBatch ) * segEnvBatch )
        shgErrs.append(loss1 )
        loss2 = torch.mean( torch.abs( renderedPred - renderedGt ) * segEnvBatch )
        shgErrs.append(loss2 )

        # Compute the loss of lightings
        totalErr = shgW * shgErrs[0] + renW * shgErrs[1]

        # Write errors to screen
        utils.writeErrToScreen('shg', shgErrs, epoch, j )

        # Write errors to file
        utils.writeErrToFile('shg', shgErrs, trainingLog, epoch, j )

        if torch.sum(torch.isnan(totalErr ).data.float() ) == 0:
            totalErr.backward()
        else:
            print(nameBatch )

        # Update the network parameter
        opIndirectLight.step()

        # Accumulate errors
        shadingErrsNpList = np.concatenate( [shadingErrsNpList, utils.turnErrorIntoNumpy( shgErrs ) ], axis=0 )

        del shgErrs
        torch.cuda.empty_cache()

        if j -opt.iterId < 1000:
            utils.writeNpErrToScreen('shadingAccu', np.mean(shadingErrsNpList[1:j+1-opt.iterId, :], axis=0), epoch, j )
            utils.writeNpErrToFile('shadingAccu', np.mean(shadingErrsNpList[1:j+1-opt.iterId, :], axis=0), trainingLog, epoch, j )
        else:
            utils.writeNpErrToScreen('shadingAccu', np.mean(shadingErrsNpList[j-999-opt.iterId:j+1-opt.iterId, :], axis=0), epoch, j )
            utils.writeNpErrToFile('shadingAccu', np.mean(shadingErrsNpList[j-999-opt.iterId:j+1-opt.iterId, :], axis=0), trainingLog, epoch, j )


        if j == 1 or j% 2000 == 0:
            # Save the groundtruth results
            vutils.save_image( ( (albedoBatch ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( ( 0.5*(normalBatch + 1) ).data,
                    '{0}/{1}_normalGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( ( 0.5*(roughBatch + 1) ).data,
                    '{0}/{1}_roughGt.png'.format(opt.experiment, j), nrow=nrow )
            depthOutGt = 1 / torch.clamp(depthBatch + 1, 1e-6, 10 )
            vutils.save_image( ( depthOutGt ).data,
                    '{0}/{1}_depthGt.png'.format(opt.experiment, j), nrow=nrow )

            # Save the predicted BRDF
            vutils.save_image( ( (albedoPred ) ** (1.0/2.2) ).data,
                    '{0}/{1}_albedoPred.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( ( 0.5*(normalPred + 1) ).data,
                    '{0}/{1}_normalPred.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( ( 0.5*(roughPred + 1) ).data,
                    '{0}/{1}_roughPred.png'.format(opt.experiment, j), nrow=nrow )

            # Output the groundtruth lighting and image
            vutils.save_image( ( (imBatch )**(1.0/2.2) ).data,
                '{0}/{1}_im.png'.format(opt.experiment, j ), nrow=nrow )

            vutils.save_image( ((shadingBatch)**(1.0/2.2) ).data,
                    '{0}/{1}_shadingGt.png'.format(opt.experiment, j), nrow = nrow )
            vutils.save_image( ((shadingDirectBatch )**(1.0/2.2) ).data,
                    '{0}/{1}_shadingDirectGt.png'.format(opt.experiment, j), nrow = nrow )

            vutils.save_image( ((renderedGt)**(1.0/2.2) ).data,
                    '{0}/{1}_renderedGt.png'.format(opt.experiment, j), nrow = nrow )

            vutils.save_image( ( (shadingPred  )**(1.0/2.2) ).data,
                '{0}/{1}_shadingPred.png'.format(opt.experiment, j), nrow=nrow )

            vutils.save_image( ( (renderedPred  )**(1.0/2.2) ).data,
                '{0}/{1}_renderedPred.png'.format(opt.experiment, j), nrow=nrow )

        if j % 5000 == 0:
            # save the models
            torch.save( {'model': indirectLightNet.module, 'optim': opIndirectLight.state_dict() },
                       '{0}/indirectLightNet_iter{1}.pth'.format(opt.experiment, j ) )

    trainingLog.close()

    # Save the error record
    np.save('{0}/shadingError_{1}.npy'.format(opt.experiment, epoch), shadingErrsNpList )

torch.save( {'model': indirectLightNet.module.state_dict(), 'optim': opIndirectLight.state_dict() },
           '{0}/indirectLightNet_iter{1}.pth'.format(opt.experiment, j ) )
