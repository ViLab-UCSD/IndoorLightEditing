import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import modelShadowDepth as modelShadow
import os.path as osp
curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
import sys
sys.path.append(osp.join(curDir, 'OptixRendererShadow/src/lib/' ) )
import scipy.ndimage  as ndimage
import utils
import os

class renderShadow():
    def __init__(self, modelRoot, iterId, isCuda = True, fov=57.95,
                 maxWinNum = 3, maxLampNum = 7,
                 winSampleNum=100, lampSampleNum = 100,
                 outputRoot = None ):

        self.denoiser = modelShadow.denoiser(fov = fov )
        modelName = osp.join(modelRoot, 'denoiser_iter%d.pth' % iterId )
        self.denoiser.load_state_dict(torch.load(modelName )['model'] )
        for param in self.denoiser.parameters():
            param.requires_grad = False
        self.denoiser = self.denoiser.cuda()

        self.fov = fov
        self.threshold = 0.12
        self.lampSampleNum = lampSampleNum
        self.winSampleNum = winSampleNum
        self.maxWinNum = maxWinNum
        self.maxLampNum = maxLampNum

        if outputRoot is None:
            self.outputRoot = curDir
        else:
            self.outputRoot = outputRoot

    def shadowToConf(self, shadow ):
        conf = (shadow < -0.005 )
        conf = ndimage.binary_dilation(conf, structure = np.ones((5, 5) ) )
        conf = conf.astype(np.float32 )
        conf[0:3, :] = 1.0
        conf[:, 0:3] = 1.0
        conf[-3:, :] = 1.0
        conf[:, -3:] = 1.0
        return conf

    def setWinNum(self, winNum ):
        self.maxWinNum = winNum

    def setLampNum(self, lampNum ):
        self.maxLampNum = lampNum

    def setOutputRoot(self, outputRoot ):
        self.outputRoot = outputRoot


    def forward(self,
                depthPred, normalPred,
                visWinCenterPreds, visWinXAxisPreds, visWinYAxisPreds, onWinMasksBatch,
                visLampCenterPreds, onLampMasksBatch,
                invWinCenterPred, invWinXAxisPred, invWinYAxisPred,
                invLampAxesPred, invLampCenterPred,
                objName=None, roomName = None, visLampMeshNames = None ):

        depthNp = depthPred.detach().cpu().numpy()

        batchSize = int(depthNp.shape[0] )
        height = int(depthNp.shape[2] )
        width = int(depthNp.shape[3] )

        if objName is None:
            objName = 'None'
        if roomName is None:
            roomName = 'None'

        import optixRenderer

        # Output meshes
        if self.maxWinNum > 0:
            utils.writeWindowList(
                    visWinCenterPreds,
                    visWinYAxisPreds,
                    visWinXAxisPreds,
                    self.maxWinNum,
                    osp.join(self.outputRoot, 'visWinPred.obj' ) )

        if self.maxLampNum  > 0:
            if visLampMeshNames is None:
                utils.writeLampList(
                        visLampCenterPreds,
                        depthPred,
                        normalPred,
                        onLampMasksBatch,
                        self.maxLampNum,
                        osp.join(self.outputRoot, 'visLampPred.ply' ) )

        utils.writeWindowBatch(
                invWinCenterPred.unsqueeze(1),
                invWinYAxisPred.unsqueeze(1),
                invWinXAxisPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                osp.join(self.outputRoot, 'invWinPred.obj' ) )

        utils.writeLampBatch(
                invLampAxesPred.unsqueeze(1),
                invLampCenterPred.unsqueeze(1),
                np.ones( (batchSize, 1 ) ),
                1,
                osp.join(self.outputRoot, 'invLampPred.ply' ) )


        # Compute shadows for visible windows
        visWinShadowInits = []
        visWinShadowPreds = []

        if len(visWinCenterPreds )  > 0:
            winMask = (onWinMasksBatch.detach().cpu().numpy() != 0 ).astype(np.float32 )
            for m in range(0, batchSize ):
                for n in range(0, self.maxWinNum ):
                    winId = m * self.maxWinNum + n
                    if visWinCenterPreds[winId ] is None:
                        visWinShadowInits.append(torch.zeros(1, 1, height, width ).cuda() )
                        visWinShadowPreds.append(torch.zeros(1, 1, height, width ).cuda() )
                    else:
                        meshName = osp.join(self.outputRoot, 'visWinPred_%d_%d.obj' % (m, n) )
                        depth = depthNp[m, 0, :, :] * (1 - winMask[m, n, :] )
                        shadow = optixRenderer.render(
                            depth.flatten(), height, width,
                            meshName,
                            self.fov, self.threshold, self.winSampleNum,
                            objName, 'None'
                        )
                        shadow = np.minimum(shadow.reshape(height, width ) + winMask[m, n, :], 1 )
                        conf = self.shadowToConf(shadow )

                        shadowInit = torch.from_numpy(shadow[np.newaxis, np.newaxis, :] ).cuda()
                        conf = torch.from_numpy(conf[np.newaxis, np.newaxis, :] ).cuda()

                        shadowPred = self.denoiser(shadowInit,
                                                     normalPred[m:m+1, :], depthPred[m:m+1, :],
                                                     conf )
                        shadowPred = torch.clamp(shadowPred + torch.from_numpy(winMask[m, n, :] ).cuda(), max=1 )

                        visWinShadowInits.append(shadowInit )
                        visWinShadowPreds.append(shadowPred )

            visWinShadowInits = torch.cat(visWinShadowInits, dim=0 ).reshape(batchSize, self.maxWinNum, 1, height, width )
            visWinShadowPreds = torch.cat(visWinShadowPreds, dim=0 ).reshape(batchSize, self.maxWinNum, 1, height, width )
            os.system('rm %s' % osp.join(self.outputRoot, 'visWinPred_*.obj') )

        # Compute shadows for visble lamps
        visLampShadowInits = []
        visLampShadowPreds = []
        if len(visLampCenterPreds ) > 0:
            lampMask = (onLampMasksBatch.detach().cpu().numpy() != 0 ).astype(np.float32 )
            for m in range(0, batchSize ):
                for n in range(0, self.maxLampNum ):
                    lampId = m * self.maxLampNum + n
                    if visLampCenterPreds[lampId ] is None:
                        visLampShadowInits.append(torch.zeros(1, 1, height, width ).cuda() )
                        visLampShadowPreds.append(torch.zeros(1, 1, height, width ).cuda() )
                    else:
                        if visLampMeshNames is None:
                            meshName = osp.join(self.outputRoot, 'visLampPred_%d_%d.obj' % (m, n) )
                        else:
                            meshName = visLampMeshNames[lampId ]

                        depth = depthNp[m, 0, :, :] * (1 - lampMask[m, n, :] )
                        shadow = optixRenderer.render(
                            depth.flatten(), height, width,
                            meshName,
                            self.fov, self.threshold, self.lampSampleNum,
                            objName, 'None'
                        )

                        shadow = shadow.reshape(height, width )
                        shadow = shadow.reshape(height, width )
                        conf = self.shadowToConf(shadow )

                        shadowInit = torch.from_numpy(shadow[np.newaxis, np.newaxis, :] ).cuda()
                        conf = torch.from_numpy(conf[np.newaxis, np.newaxis, :] ).cuda()

                        shadowPred = self.denoiser(shadowInit,
                                                     normalPred[m:m+1, :], depthPred[m:m+1, :],
                                                     conf )
                        shadowPred = torch.clamp(shadowPred + torch.from_numpy(lampMask[m, n, :] ).cuda(), max=1 )

                        visLampShadowInits.append(shadowInit )
                        visLampShadowPreds.append(shadowPred )

            visLampShadowInits = torch.cat(visLampShadowInits, dim=0 ).reshape(batchSize, self.maxLampNum, 1, height, width )
            visLampShadowPreds = torch.cat(visLampShadowPreds, dim=0 ).reshape(batchSize, self.maxLampNum, 1, height, width )

            os.system('rm %s' % osp.join(self.outputRoot, 'visLampPred_*.obj') )

        # Compute shadow for invisible window
        invWinShadowInit = []
        invWinShadowPred = []
        for m in range(0, batchSize ):
            meshName = osp.join(self.outputRoot, 'invWinPred_%d_%d.obj' % (m, 0) )
            depth = depthNp[m, 0, :, :]
            shadow = optixRenderer.render(
                depth.flatten(), height, width,
                meshName,
                self.fov, self.threshold, self.winSampleNum,
                objName, 'None'
            )
            shadow = shadow.reshape(height, width )
            conf = self.shadowToConf(shadow )

            shadowInit = torch.from_numpy(shadow[np.newaxis, np.newaxis, :] ).cuda()
            conf = torch.from_numpy(conf[np.newaxis, np.newaxis, :] ).cuda()

            shadowPred = self.denoiser(shadowInit,
                                         normalPred[m:m+1, :], depthPred[m:m+1, :],
                                         conf )
            invWinShadowInit.append(shadowInit )
            invWinShadowPred.append(shadowPred )

        invWinShadowInit = torch.cat(invWinShadowInit, dim=0 )
        invWinShadowPred = torch.cat(invWinShadowPred, dim=0 )

        # Compute shadow for invisible lamp
        invLampShadowInit = []
        invLampShadowPred = []
        for m in range(0, batchSize ):
            meshName = osp.join(self.outputRoot, 'invLampPred_%d_%d.obj' % (m, 0) )
            depth = depthNp[m, 0, :, :]
            if m == 0:
                shadow = optixRenderer.render(
                    depth.flatten(), height, width,
                    meshName,
                    self.fov, self.threshold, self.lampSampleNum,
                    objName, roomName
                )
            else:
                shadow = optixRenderer.render(
                    depth.flatten(), height, width,
                    meshName,
                    self.fov, self.threshold, self.lampSampleNum,
                    objName, 'None'
                )

            shadow = shadow.reshape(height, width )
            conf = self.shadowToConf(shadow )

            shadowInit = torch.from_numpy(shadow[np.newaxis, np.newaxis, :] ).cuda()
            conf = torch.from_numpy(conf[np.newaxis, np.newaxis, :] ).cuda()

            shadowPred = self.denoiser(shadowInit,
                                         normalPred[m:m+1, :], depthPred[m:m+1, :],
                                         conf )
            invLampShadowInit.append(shadowInit )
            invLampShadowPred.append(shadowPred )

        invLampShadowInit = torch.cat(invLampShadowInit, dim=0 )
        invLampShadowPred = torch.cat(invLampShadowPred, dim=0 )

        os.system('rm %s' % osp.join(self.outputRoot, 'invWinPred_*.obj') )
        os.system('rm %s' % osp.join(self.outputRoot, 'invLampPred_*.obj') )

        del optixRenderer

        return visWinShadowInits, visWinShadowPreds, \
            visLampShadowInits, visLampShadowPreds, \
            invWinShadowInit, invWinShadowPred, \
            invLampShadowInit, invLampShadowPred
