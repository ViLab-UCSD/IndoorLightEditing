import torch
import numpy as np
import torch.optim as optim
import argparse
import random
import os
import models
import torchvision.utils as vutils
import utils
import dataLoader
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os.path as osp
import lossFunctions

parser = argparse.ArgumentParser()
# The locationi of training set
parser.add_argument('--dataRoot',default='Dataset', help='path to input images')
parser.add_argument('--experiment', default=None, help='the path to store samples and models')
# The basic training setting
parser.add_argument('--nepoch', type=int, default=15, help='the number of epochs for training')
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--imHeight', type=int, default=240, help='the height / width of the input image to network')
parser.add_argument('--imWidth', type=int, default=320, help='the height / width of the input image to network')
parser.add_argument('--deviceIds', type=int, nargs='+', default=[0], help='the gpus used for training network')
# Fine tune the network
parser.add_argument('--isFineTune', action='store_true', help='fine-tune the network')
parser.add_argument('--epochId', type=int, default = -1, help='the training of epoch of the loaded model')
# The training weight
parser.add_argument('--albedoWeight', type=float, default=1.5, help='the weight for the diffuse component')
parser.add_argument('--normalWeight', type=float, default=1.0, help='the weight for the diffuse component')
parser.add_argument('--roughWeight', type=float, default=0.5, help='the weight for the roughness component')

# The detail network setting
opt = parser.parse_args()
print(opt)

albeW, normW = opt.albedoWeight, opt.normalWeight
rougW = opt.roughWeight

curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
if opt.experiment is None:
    opt.experiment = 'check_brdf_w%d_h%d' % (opt.imWidth, opt.imHeight )

opt.experiment = osp.join(curDir, opt.experiment )
os.system('mkdir {0}'.format(opt.experiment) )
os.system('cp *.py %s' % opt.experiment )

opt.seed = 0
print("Random Seed: ", opt.seed )
random.seed(opt.seed )
torch.manual_seed(opt.seed )

# Initial Network
encoder = models.encoder(isGtGeometry = True )
albedoDecoder = models.decoder(mode=0 )
normalDecoder = models.decoder(mode=1 )
roughDecoder = models.decoder(mode=2 )

if opt.isFineTune:
    encoder.load_state_dict(
            torch.load('{0}/encoder_{1}.pth'.format(opt.experiment, opt.epochId ) ) )
    albedoDecoder.load_state_dict(
            torch.load('{0}/albedo_{1}.pth'.format(opt.experiment, opt.epochId ) ) )
    normalDecoder.load_state_dict(
            torch.load('{0}/normal_{1}.pth'.format(opt.experiment, opt.epochId ) ) )
    roughDecoder.load_state_dict(
            torch.load('{0}/rough_{1}.pth'.format(opt.experiment, opt.epochId ) ) )

encoder = nn.DataParallel(encoder, device_ids = opt.deviceIds )
albedoDecoder = nn.DataParallel(albedoDecoder, device_ids = opt.deviceIds )
normalDecoder = nn.DataParallel(normalDecoder, device_ids = opt.deviceIds )
roughDecoder = nn.DataParallel(roughDecoder, device_ids = opt.deviceIds )

encoder = encoder.cuda()
albedoDecoder = albedoDecoder.cuda()
normalDecoder = normalDecoder.cuda()
roughDecoder = roughDecoder.cuda()

# Optimizer
opEncoder = optim.Adam(encoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )
opAlbedo = optim.Adam(albedoDecoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )
opNormal = optim.Adam(normalDecoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )
opRough = optim.Adam(roughDecoder.parameters(), lr=1e-4, betas=(0.5, 0.999) )

brdfDataset = dataLoader.BatchLoader( opt.dataRoot,
        imWidth = opt.imWidth, imHeight = opt.imHeight )
brdfLoader = DataLoader(brdfDataset, batch_size = opt.batchSize,
        num_workers = 8, shuffle = True )

j = 0
albedoErrsNpList = np.ones( [1, 1], dtype = np.float32 )
normalErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughErrsNpList= np.ones( [1, 1], dtype = np.float32 )
albedoGradErrsNpList = np.ones( [1, 1], dtype = np.float32 )
roughGradErrsNpList= np.ones( [1, 1], dtype = np.float32 )

nrow = opt.batchSize  * 2

for epoch in list(range(opt.epochId + 1, opt.nepoch ) ):
    trainingLog = open('{0}/trainingLog_{1}.txt'.format(opt.experiment, epoch), 'w' )
    for i, dataBatch in enumerate(brdfLoader):
        j += 1
        # Load data from cpu to gpu
        albedoBatch, normalBatch, roughBatch, \
                depthBatch, depthOriginBatch, depthScaleBatch, \
                segBRDFBatch, segAllBatch \
                = dataLoader.getBRDF(dataBatch )

        # Load the image from cpu to gpu
        im_cpu = dataBatch['im']
        imBatch = im_cpu.cuda()
        imDl_cpu = dataBatch['imDl']
        imDlBatch = imDl_cpu.cuda()
        imDm_cpu = dataBatch['imDm']
        imDmBatch = imDm_cpu.cuda()
        imBatch = torch.cat([imBatch, imDlBatch, imDmBatch], dim=0 )

        # Clear the gradient in optimizer
        opEncoder.zero_grad()
        opAlbedo.zero_grad()
        opNormal.zero_grad()
        opRough.zero_grad()

        # Build the cascade network architecture #
        depthMax = torch.max(torch.max(depthBatch, dim=2, keepdim=True )[0], dim=3, keepdim=True )[0]
        depthBatch = depthBatch * segAllBatch + (1 - segAllBatch ) * depthMax
        inputBatch = torch.cat([imBatch, depthBatch], dim=1 )

        # Initial Prediction
        x1, x2, x3, x4, x5, x6 = encoder(inputBatch )
        albedoPred, _ = albedoDecoder(x1, x2, x3,
                x4, x5, x6, [opt.imHeight, opt.imWidth ] )
        normalPred, _ = normalDecoder(x1, x2, x3,
                x4, x5, x6, [opt.imHeight, opt.imWidth] )
        roughPred, _ = roughDecoder(x1, x2, x3,
                x4, x5, x6, [opt.imHeight, opt.imWidth] )

        # Compute the error
        albedoErrs = []
        normalErrs = []
        roughErrs = []
        albedoGradErrs = []
        roughGradErrs = []

        albedoErrs.append( torch.mean( torch.pow(albedoPred - albedoBatch, 2) * segBRDFBatch ) )
        normalErrs.append( torch.mean( torch.pow(normalPred - normalBatch, 2) * segBRDFBatch ) )
        roughErrs.append( torch.mean( torch.pow(roughPred - roughBatch, 2) * segBRDFBatch ) )
        albedoGradErrs.append(lossFunctions.gradLoss(albedoPred, albedoBatch, segBRDFBatch ) )
        roughGradErrs.append(lossFunctions.gradLoss(roughPred, roughBatch, segBRDFBatch ) )

        # Back propagate the gradients
        totalErr = albeW * sum(albedoErrs ) + normW * sum(normalErrs ) \
                + rougW * sum(roughErrs )
        totalErr += albeW * sum(albedoGradErrs ) + rougW * sum(roughGradErrs )

        totalErr.backward()

        # Update the network parameter
        opEncoder.step()
        opAlbedo.step()
        opNormal.step()
        opRough.step()

        # Output training error
        utils.writeErrToScreen('albedo', albedoErrs, epoch, j )
        utils.writeErrToScreen('normal', normalErrs, epoch, j )
        utils.writeErrToScreen('rough', roughErrs, epoch, j )

        utils.writeErrToFile('albedo', albedoErrs, trainingLog, epoch, j )
        utils.writeErrToFile('normal', normalErrs, trainingLog, epoch, j )
        utils.writeErrToFile('rough', roughErrs, trainingLog, epoch, j )
        utils.writeErrToFile('albedoGrad', albedoGradErrs, trainingLog, epoch, j )
        utils.writeErrToFile('roughGrad', roughGradErrs, trainingLog, epoch, j )

        albedoErrsNpList = np.concatenate( [albedoErrsNpList, utils.turnErrorIntoNumpy(albedoErrs)], axis=0 )
        normalErrsNpList = np.concatenate( [normalErrsNpList, utils.turnErrorIntoNumpy(normalErrs)], axis=0 )
        roughErrsNpList = np.concatenate( [roughErrsNpList, utils.turnErrorIntoNumpy(roughErrs)], axis=0 )
        albedoGradErrsNpList = np.concatenate( [albedoGradErrsNpList, utils.turnErrorIntoNumpy(albedoGradErrs)], axis=0 )
        roughGradErrsNpList = np.concatenate( [roughGradErrsNpList, utils.turnErrorIntoNumpy(roughGradErrs)], axis=0 )


        if j < 1000:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('albedoGradAccu', np.mean(albedoGradErrsNpList[1:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughGradAccu', np.mean(roughGradErrsNpList[1:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('albedoGradAccu', np.mean(albedoGradErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('roughGradAccu', np.mean(roughGradErrsNpList[1:j+1, :], axis=0), trainingLog, epoch, j )
        else:
            utils.writeNpErrToScreen('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('albedoGradAccu', np.mean(albedoGradErrsNpList[j-999:j+1, :], axis=0), epoch, j )
            utils.writeNpErrToScreen('roughGradAccu', np.mean(roughGradErrsNpList[j-999:j+1, :], axis=0), epoch, j )

            utils.writeNpErrToFile('albedoAccu', np.mean(albedoErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('normalAccu', np.mean(normalErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('roughAccu', np.mean(roughErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('albedoGradAccu', np.mean(albedoGradErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )
            utils.writeNpErrToFile('roughGradAccu', np.mean(roughGradErrsNpList[j-999:j+1, :], axis=0), trainingLog, epoch, j )


        if j == 1 or j% 2000 == 0:
            # Save the ground truth and the input
            vutils.save_image( albedoBatch ** (1.0/2.2),
                    '{0}/{1}_albedoGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( 0.5*(normalBatch + 1),
                    '{0}/{1}_normalGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( roughBatch,
                    '{0}/{1}_roughGt.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( imBatch**(1.0/2.2),
                    '{0}/{1}_im.png'.format(opt.experiment, j), nrow=nrow )
            depthOut = 1 / torch.clamp(depthBatch + 1, 1e-6, 10) * segAllBatch
            vutils.save_image( depthOut*segAllBatch,
                    '{0}/{1}_depthGt.png'.format(opt.experiment, j), nrow=nrow )

            # Save the predicted results
            vutils.save_image( albedoPred ** (1.0/2.2),
                    '{0}/{1}_albedoPred.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image( 0.5*(normalPred + 1),
                    '{0}/{1}_normalPred.png'.format(opt.experiment, j), nrow=nrow )
            vutils.save_image(roughPred,
                    '{0}/{1}_roughPred.png'.format(opt.experiment, j), nrow=nrow )

    trainingLog.close()

    # Save the error record
    np.save('{0}/albedoError_{1}.npy'.format(opt.experiment, epoch), albedoErrsNpList )
    np.save('{0}/normalError_{1}.npy'.format(opt.experiment, epoch), normalErrsNpList )
    np.save('{0}/roughError_{1}.npy'.format(opt.experiment, epoch), roughErrsNpList  )
    np.save('{0}/albedoGradError_{1}.npy'.format(opt.experiment, epoch), albedoGradErrsNpList )
    np.save('{0}/roughGradError_{1}.npy'.format(opt.experiment, epoch), roughGradErrsNpList )

    # save the models
    torch.save(encoder.module.state_dict(), '{0}/encoder_{1}.pth'.format(opt.experiment, epoch ) )
    torch.save(albedoDecoder.module.state_dict(), '{0}/albedo_{1}.pth'.format(opt.experiment, epoch ) )
    torch.save(normalDecoder.module.state_dict(), '{0}/normal_{1}.pth'.format(opt.experiment, epoch ) )
    torch.save(roughDecoder.module.state_dict(), '{0}/rough_{1}.pth'.format(opt.experiment, epoch ) )
