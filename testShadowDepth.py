import utils
import torch
import numpy as np
import argparse
import random
import os
import models
import torchvision.utils as vutils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
import lossFunctions
import modelShadowDepth as modelShadow
import scipy.ndimage as ndimage

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images')
parser.add_argument('--experimentBRDF', default=None, help='the path to store samples and models' )
parser.add_argument('--experiment', default=None, help='the path to store samples and models' )
parser.add_argument('--testRoot', default=None, help='the path to test results')
# The basic training setting
parser.add_argument('--nepochBRDF', type=int, default=15, help='the number of epochs for BRDF prediction')
parser.add_argument('--nepoch', type=int, default=2, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size' )
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--shadowRow', type=int, default=120, help='the number of samples of envmaps in y direction')
parser.add_argument('--shadowCol', type=int, default=160, help='the number of samples of envmaps in x direction')
parser.add_argument('--fov', type=int, default=57.95, help='the field of view when capturing image')

parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')

# Fine-tune the results
parser.add_argument('--iterId', default = 70000, type=int, help='the id of epoch we use for fine-tuneing')

# Training setting
parser.add_argument('--isGradLoss', action='store_true', help='whether to add gradient loss')

# The training weight
parser.add_argument('--shadowWeight', type=float, default=1.0, help='the weight for lambda of visible light sources')
# The detail network setting
opt = parser.parse_args()
print(opt )

opt.gpuId = opt.deviceIds[0]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.autograd.set_detect_anomaly(True )

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_shadowDepth'
    if opt.isGradLoss:
        opt.experiment += '_grad'

opt.experiment = osp.join(curDir, opt.experiment )
if opt.testRoot is None:
    opt.testRoot = opt.experiment.replace('check_', 'test_')
os.system('mkdir {0}'.format(opt.testRoot ) )
os.system('cp %s/*.py %s' % (curDir, opt.testRoot ) )


opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )


# Network for BRDF prediction
if opt.experimentBRDF is None:
    opt.experimentBRDF = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )
opt.experimentBRDF = osp.join(curDir, opt.experimentBRDF )

encoder = models.encoder(isGtGeometry = True )
normalDecoder = models.decoder(mode=1 )

encoder.load_state_dict(
    torch.load('{0}/encoder_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )
normalDecoder.load_state_dict(
    torch.load('{0}/normal_{1}.pth'.format(opt.experimentBRDF, opt.nepochBRDF-1 ) ) )

for param in encoder.parameters():
    param.requires_grad = False
for param in normalDecoder.parameters():
    param.requires_grad = False

encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )

denoiser = modelShadow.denoiser()
denoiserDict = torch.load('{0}/denoiser_iter{1}.pth'.format(opt.experiment, opt.iterId ) )
denoiser.load_state_dict(denoiserDict['model'] )
for para in denoiser.parameters():
    para.requires_grad = False

denoiser = nn.DataParallel(denoiser, device_ids = opt.deviceIds )

encoder = encoder.cuda()
normalDecoder = normalDecoder.cuda()
denoiser = denoiser.cuda()

####################################
brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight,
        isShading = False, isLightSrc = True, isDepthShadow = True, phase = 'TEST' )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 0, shuffle = False, drop_last = False )

j = 0
shadowErrsNpList = np.ones( [1, 2], dtype = np.float32 )

testingLog = open('{0}/testingLog_iter{1}.txt'.format(opt.testRoot, opt.iterId ), 'w')
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

    semLabel_cpu = dataBatch['semLabel']
    semLabelBatch = semLabel_cpu.cuda()

    # Load visible window
    visWinNum, winMasksBatch, onWinMasksBatch, \
            visWinPlanesBatch, visWinSrcsBatch, \
            visWinShadingsBatch, visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
            visWinShadowsBatch, depthVisWinShadowsBatch, confVisWinShadowsBatch \
            = dataLoader.getVisWindow(dataBatch, isDepthShadow = True )

    # Load visible lamp
    visLampNum, lampMasksBatch, onLampMasksBatch, \
            visLampCentersBatch, visLampAxesBatch, \
            visLampShadingsBatch, visLampShadingsNoBatch, \
            visLampShadowsBatch, depthVisLampShadowsBatch, confVisLampShadowsBatch \
            = dataLoader.getVisLamp(dataBatch, isDepthShadow = True )

    # Load invisible window
    invWinNum, invWinOnBatch, \
            invWinPlanesBatch, invWinSrcsBatch, \
            invWinShadingsBatch, invWinShadingsNoBatch, invWinShadingsNoAppBatch, \
            invWinShadowsBatch, depthInvWinShadowsBatch, confInvWinShadowsBatch \
            = dataLoader.getInvWindow(dataBatch, isDepthShadow = True )

    # Load invisible lamp
    invLampNum, invLampOnBatch, \
            invLampCentersBatch, invLampAxesBatch, \
            invLampShadingsBatch, \
            invLampShadingsNoBatch, invLampShadowsBatch, \
            depthInvLampShadowsBatch, confInvLampShadowsBatch \
            = dataLoader.getInvLamp(dataBatch, isDepthShadow = True )

    visWinNum = int(visWinNum[0].squeeze() )
    visLampNum = int(visLampNum[0].squeeze() )
    invWinNum = int(invWinNum[0].squeeze() )
    invLampNum = int(invLampNum[0].squeeze() )

    lightNum = visWinNum + visLampNum + invWinNum + invLampNum

    nameBatch = dataBatch['name']
    batchSize = imBatch.size(0 )

    depthMax = torch.max(torch.max(depthBatch, dim=2, keepdim=True )[0], dim=3, keepdim=True )[0]
    depthBatch = depthBatch * segAllBatch + (1 - segAllBatch ) * depthMax
    inputBatch = torch.cat([imBatch, depthBatch], dim=1 )

    # Predict the large BRDF
    x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
    normalPred, nf = normalDecoder(x1, x2, x3,
            x4, x5, x6, [opt.imHeight, opt.imWidth] )
    depthPred = depthBatch

    imBatch = F.adaptive_avg_pool2d(imBatch, (opt.shadowRow, opt.shadowCol ) )
    normalPred = F.adaptive_avg_pool2d(normalPred, (opt.shadowRow, opt.shadowCol ) )
    depthPred =  F.adaptive_avg_pool2d(depthPred, (opt.shadowRow, opt.shadowCol ) )
    segBRDFBatch = F.adaptive_avg_pool2d(segBRDFBatch, (opt.shadowRow, opt.shadowCol ) )

    # Combine the light sources
    depthShadows = []
    confs = []
    lightMasks = []
    shadowGts = []
    depthPreds = []
    normalPreds = []
    segBRDFs = []

    if visLampNum  > 0:
        shadowGts.append(visLampShadowsBatch[0, 0:visLampNum] )
        lightMasks.append(lampMasksBatch[0, 0:visLampNum ].unsqueeze(1) )
        depthShadows.append(depthVisLampShadowsBatch[0, 0:visLampNum ] )
        confs.append(confVisLampShadowsBatch[0, 0:visLampNum ] )
        for n in range(0, visLampNum ):
            depthPreds.append(depthPred[0:1, :] )
            normalPreds.append(normalPred[0:1, :] )

    if invLampNum > 0:
        shadowGts.append(invLampShadowsBatch[0, 0:invLampNum ] )
        lightMasks.append(torch.zeros([invLampNum, 1, opt.imHeight, opt.imWidth ],
                                     dtype=torch.float32 ).cuda() )
        depthShadows.append(depthInvLampShadowsBatch[0, 0:invLampNum ] )
        confs.append(confInvLampShadowsBatch[0, 0:invLampNum ] )
        for n in range(0, invLampNum ):
            depthPreds.append(depthPred[0:1, :] )
            normalPreds.append(normalPred[0:1, :] )

    if visWinNum > 0:
        shadowGts.append(visWinShadowsBatch[0, 0:visWinNum ] )
        lightMasks.append(winMasksBatch[0, 0:visWinNum].unsqueeze(1) )
        depthShadows.append(depthVisWinShadowsBatch[0, 0:visWinNum ] )
        confs.append(confVisWinShadowsBatch[0, 0:visWinNum ] )
        for n in range(0, visWinNum ):
            depthPreds.append(depthPred[0:1, :] )
            normalPreds.append(normalPred[0:1, :] )

    if invWinNum > 0:
        shadowGts.append(invWinShadowsBatch[0, 0:invWinNum] )
        lightMasks.append(torch.zeros([invWinNum, 1, opt.imHeight, opt.imWidth ],
                                     dtype=torch.float32 ).cuda() )
        depthShadows.append(depthInvWinShadowsBatch[0, 0:invWinNum] )
        confs.append(confInvWinShadowsBatch[0, 0:invWinNum ] )
        for n in range(0, invWinNum ):
            depthPreds.append(depthPred[0:1, :] )
            normalPreds.append(normalPred[0:1, :] )

    lightMasks = torch.cat(lightMasks, dim=0 )
    lightMask = torch.sum(lightMasks, dim=0, keepdim=True )
    lightMask = torch.clamp(1 - lightMask, 0, 1 )
    lightMask = F.adaptive_avg_pool2d(lightMask, (opt.shadowRow, opt.shadowCol ) )

    semLabelSmallBatch = F.adaptive_avg_pool2d(semLabelBatch, (opt.shadowRow, opt.shadowCol ) )
    semLabelSmallBatch = (semLabelSmallBatch > 0.999).float()

    lightMask = lightMask  * semLabelSmallBatch

    shadowGts = torch.cat(shadowGts, dim=0 )
    depthShadows = torch.cat(depthShadows, dim=0 )
    confs = torch.cat(confs, dim=0 )
    depthPreds = torch.cat(depthPreds, dim=0 )
    normalPreds = torch.cat(normalPreds, dim=0 )

    shadowPreds = denoiser(depthShadows, normalPreds, depthPreds, confs )
    shadowPredsScaled, _ = models.LSregress(
        shadowPreds.detach() * lightMask,
        shadowGts * lightMask,
        shadowPreds
    )
    depthShadowsScaled, _ = models.LSregress(
        torch.clamp(depthShadows, 0, 1) * lightMask,
        shadowGts * lightMask,
        torch.clamp(depthShadows, 0, 1)
    )

    shadowErr = torch.mean(torch.pow(shadowPredsScaled - shadowGts, 2) * lightMask )
    shadowOrigErr = torch.mean(torch.pow(depthShadowsScaled - shadowGts, 2) * lightMask )

    # Output training error
    utils.writeErrToScreen('shadow', [shadowOrigErr, shadowErr ], opt.iterId, j )
    utils.writeErrToFile('shadow', [shadowOrigErr, shadowErr ], testingLog, opt.iterId, j )
    shadowErrsNpList = np.concatenate( [shadowErrsNpList, utils.turnErrorIntoNumpy([shadowOrigErr, shadowErr ] ) ], axis=0 )

    utils.writeNpErrToScreen('shadowAccu', np.mean(shadowErrsNpList[1:j+1, :], axis=0), opt.iterId, j )
    utils.writeNpErrToFile('shadowAccu', np.mean(shadowErrsNpList[1:j+1, :], axis=0), testingLog, opt.iterId, j )

    if j == 1 or j% 200 == 0:
        # Save the ground truth and the input
        vutils.save_image( ( (imBatch)**(1.0/2.2) ).data,
                '{0}/{1}_im.png'.format(opt.testRoot, j), nrow=batchSize )
        vutils.save_image( (0.5*(normalBatch + 1) ).data,
                '{0}/{1}_normalGt.png'.format(opt.testRoot, j), nrow=batchSize )
        depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10)
        vutils.save_image( depthOut.data,
                '{0}/{1}_depthGt.png'.format(opt.testRoot, j), nrow=batchSize )
        vutils.save_image( shadowGts.data,
                '{0}/{1}_shadowGt.png'.format(opt.testRoot, j), nrow=batchSize )

        # Save the predicted results
        vutils.save_image( ( 0.5*(normalPred + 1) ).data,
                '{0}/{1}_normalPred.png'.format(opt.testRoot, j ), nrow=batchSize )

        vutils.save_image( depthShadows.data,
                '{0}/{1}_shadowOrigPred.png'.format(opt.testRoot, j), nrow=lightNum )
        vutils.save_image( depthShadowsScaled.data,
                '{0}/{1}_shadowOrigPredScaled.png'.format(opt.testRoot, j), nrow=lightNum )
        vutils.save_image( shadowPreds.data,
                '{0}/{1}_shadowPred.png'.format(opt.testRoot, j), nrow=lightNum )
        vutils.save_image( shadowPredsScaled.data,
                '{0}/{1}_shadowPredScaled.png'.format(opt.testRoot, j), nrow=lightNum )

        vutils.save_image( confs.data,
                '{0}/{1}_confidence.png'.format(opt.testRoot, j), nrow=lightNum )


testingLog.close()

# Save the error record
np.save('{0}/shadowError_iter{1}.npy'.format(opt.testRoot, opt.iterId ), shadowErrsNpList )
