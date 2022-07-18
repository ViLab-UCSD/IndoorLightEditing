import utils
import torch
import numpy as np
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
import lossFunctions

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images' )
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--testRoot', default=None, help='the path to test results' )
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction' )
parser.add_argument('--nepoch', type=int, default=6, help='the number of epochs for training' )
parser.add_argument('--batchSize', type=int, default=1, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network' )
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network' )
parser.add_argument('--envRow', type=int, default=120, help='the number of samples of envmaps in y direction' )
parser.add_argument('--envCol', type=int, default=160, help='the number of samples of envmaps in x direction' )
parser.add_argument('--envHeight', type=int, default=8, help='the size of envmaps in y direction' )
parser.add_argument('--envWidth', type=int, default=16, help='the size of envmaps in x direction' )

parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')

# Fine-tune the results
parser.add_argument('--iterId', default=240000, type=int, help='the id of epoch we use for fine-tuneing' )

# The training weight
parser.add_argument('--lightWeight', type=float, default=1.0, help='the weight for the diffuse component' )
parser.add_argument('--shadingWeight', type=float, default=0.01, help='the weight for rendering error' )

# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_shadingToLight'
opt.experiment = osp.join(curDir, opt.experiment )

if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check_', 'test_')
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

# Network for BRDF prediction
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

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

lightEncoder = modelLight.encoderLight()
weightDecoder = modelLight.decoderLight(mode = 2 )
axisDecoder = modelLight.decoderLight(mode = 0 )
lambDecoder = modelLight.decoderLight(mode = 1 )

lightEncoderDict = torch.load('{0}/lightEncoder_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
weightDecoderDict = torch.load('{0}/weightDecoder_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
axisDecoderDict = torch.load('{0}/axisDecoder_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
lambDecoderDict = torch.load('{0}/lambDecoder_iter{1}.pth'.format(opt.experiment, opt.iterId ) )

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

lightEncoder = nn.DataParallel(lightEncoder, device_ids = opt.deviceIds )
weightDecoder = nn.DataParallel(weightDecoder, device_ids = opt.deviceIds )
axisDecoder = nn.DataParallel(axisDecoder, device_ids = opt.deviceIds )
lambDecoder = nn.DataParallel(lambDecoder, device_ids = opt.deviceIds )

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

encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
normalDecoder = normalDecoder.cuda()
roughDecoder = roughDecoder.cuda()

lightEncoder = lightEncoder.cuda()
weightDecoder = weightDecoder.cuda()
axisDecoder = axisDecoder.cuda()
lambDecoder = lambDecoder.cuda()

envWeight = envWeight.cuda()

####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight, isLight = True,
        isShading = True, isLightSrc = True, phase = 'TEST')
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 6, shuffle = False, drop_last = False )

j = 0
lightErrsNpList = np.ones( [1, 1], dtype = np.float32 )
shadingErrsNpList = np.ones( [1, 2], dtype = np.float32 )

nrow = opt.batchSize * 3

testingLog = open('{0}/testingLog_iter{1}.txt'.format(opt.testRoot, opt.iterId), 'w' )
for i, dataBatch in enumerate(brdfLoader ):
    j += 1
    # Load brdf
    albedoBatch, normalBatch, roughBatch, \
            depthBatch, depthOriginBatch, depthScaleBatch, \
            segBRDFBatch, segAllBatch \
            = dataLoader.getBRDF(dataBatch )

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

    # Load environment maps
    shadingBatch, shadingDirectBatch \
            = dataLoader.getShading(dataBatch )

    # Build the cascade network architecture #
    depthMax = torch.max(torch.max(depthBatch, dim=2, keepdim=True )[0], dim=3, keepdim=True )[0]
    depthBatch = depthBatch * segAllBatch + (1 - segAllBatch ) * depthMax
    inputBatch = torch.cat([imBatch, depthBatch], dim=1 )

    # BRDF Prediction
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    albedoPred, _ = albedoDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth ] )
    normalPred, _ = normalDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth] )
    roughPred, _ = roughDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth] )

    albedoDS = F.adaptive_avg_pool2d(albedoPred, (opt.envRow, opt.envCol ) )
    normalDS = F.adaptive_avg_pool2d(normalPred, (opt.envRow, opt.envCol ) )
    roughDS = F.adaptive_avg_pool2d(roughPred, (opt.envRow, opt.envCol ) )
    depthDS = F.adaptive_avg_pool2d(depthBatch, (opt.envRow, opt.envCol ) )

    segEnvBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.envRow, opt.envCol ) )
    imSmallBatch = F.adaptive_avg_pool2d(imBatch, (opt.envRow, opt.envCol ) )
    albedoSmallBatch = F.adaptive_avg_pool2d(albedoBatch, (opt.envRow, opt.envCol ) )
    lightOnMasksSmallBatch = F.adaptive_avg_pool2d(lightOnMasksBatch, (opt.envRow, opt.envCol ) )

    segEnvBatch = (segEnvBatch == 1).float()

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

    shadingPred = utils.envToShading(envPred )

    lightErr = torch.mean(
        torch.pow(torch.log(envPred + 1 ) - torch.log(envBatch + 1 ),
                  2 ) * segEnvBatch.unsqueeze(-1).unsqueeze(-1) * envWeight )

    shgErr = torch.mean( torch.abs(shadingPred - shadingBatch) * segEnvBatch )
    renErr = torch.mean( torch.abs(shadingPred * albedoDS - shadingBatch * albedoSmallBatch ) * segEnvBatch )

    # Write errors to screen
    utils.writeErrToScreen('shg', [shgErr, renErr ], opt.iterId, j )
    utils.writeErrToScreen('light', [lightErr ], opt.iterId, j )

    utils.writeErrToFile('shg', [shgErr, renErr ], testingLog, opt.iterId, j )
    utils.writeErrToFile('light', [lightErr ], testingLog, opt.iterId, j )

    # Accumulate errors
    shadingErrsNpList = np.concatenate( [ shadingErrsNpList, utils.turnErrorIntoNumpy( [shgErr, renErr ] ) ], axis=0 )
    lightErrsNpList = np.concatenate( [ lightErrsNpList, utils.turnErrorIntoNumpy( [lightErr ] ) ], axis=0 )

    # Write errors to screen
    utils.writeNpErrToScreen('shadingAccu', np.mean(shadingErrsNpList[1:j+1, :], axis=0), opt.iterId, j )
    utils.writeNpErrToScreen('lightAccu', np.mean(lightErrsNpList[1:j+1, :], axis=0), opt.iterId, j )

    # Write errors to file
    utils.writeNpErrToFile('shadingAccu', np.mean(shadingErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )
    utils.writeNpErrToFile('lightAccu', np.mean(lightErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )

    if j == 1 or j% 200 == 0:
        vutils.save_image( albedoBatch ** (1.0/2.2),
                '{0}/{1}_albedoGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( 0.5*(normalBatch + 1),
                '{0}/{1}_normalGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( roughBatch,
                '{0}/{1}_roughGt.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( imBatch**(1.0/2.2),
                '{0}/{1}_im.png'.format(opt.testRoot, j), nrow=nrow )
        depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10) * segAllBatch
        vutils.save_image( depthOut*segAllBatch,
                '{0}/{1}_depthGt.png'.format(opt.testRoot, j), nrow=nrow )

        # Save the predicted results
        vutils.save_image( albedoPred ** (1.0/2.2),
                '{0}/{1}_albedoPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image( 0.5*(normalPred + 1),
                '{0}/{1}_normalPred.png'.format(opt.testRoot, j), nrow=nrow )
        vutils.save_image(roughPred,
                '{0}/{1}_roughPred.png'.format(opt.testRoot, j), nrow=nrow )

        vutils.save_image( ( (shadingBatch )**(1.0/2.2) ).data,
            '{0}/{1}_shadingGt.png'.format(opt.testRoot, j), nrow=nrow )
        utils.writeEnvToFile(envBatch, nrow, '{0}/{1}_envmapGt.hdr'.format(opt.testRoot, j) )

        vutils.save_image( ( (shadingPred  )**(1.0/2.2) ).data,
            '{0}/{1}_shadingPred.png'.format(opt.testRoot, j), nrow=nrow )
        utils.writeEnvToFile(envPred, nrow, '{0}/{1}_envmapPred.hdr'.format(opt.testRoot, j ) )

testingLog.close()
# Save the error record
np.save('{0}/shadingError_iter{1}.npy'.format(opt.testRoot, opt.iterId ), shadingErrsNpList )
np.save('{0}/lightError_iter{1}.npy'.format(opt.testRoot, opt.iterId ), lightErrsNpList )
