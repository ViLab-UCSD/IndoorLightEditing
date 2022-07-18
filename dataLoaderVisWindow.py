import glob
import time
import numpy as np
import os.path as osp
from PIL import Image
import random
import struct
from torch.utils.data import Dataset
import scipy.ndimage as ndimage
import cv2
from skimage.measure import block_reduce
import h5py
import scipy.ndimage as ndimage
import pickle
import torch

def getBRDF(dataBatch ):
    # Load BRDF
    albedo_cpu = dataBatch['albedo']
    albedoBatch = albedo_cpu.cuda()
    albedoDm_cpu = dataBatch['albedoDm']
    albedoDmBatch = albedoDm_cpu.cuda()
    albedoBatch = torch.cat([albedoBatch,
        albedoBatch, albedoDmBatch ], dim=0 )

    normal_cpu = dataBatch['normal']
    normalBatch = normal_cpu.cuda()
    normalDm_cpu = dataBatch['normalDm']
    normalDmBatch = normalDm_cpu.cuda()
    normalBatch = torch.cat([normalBatch,
        normalBatch, normalDmBatch ], dim=0 )

    rough_cpu = dataBatch['rough']
    roughBatch = rough_cpu.cuda()
    roughDm_cpu = dataBatch['roughDm']
    roughDmBatch = roughDm_cpu.cuda()
    roughBatch = torch.cat([roughBatch,
        roughBatch, roughDmBatch ], dim=0 )

    depth_cpu = dataBatch['depth']
    depthBatch = depth_cpu.cuda()
    depthBatch = torch.cat([depthBatch,
        depthBatch, depthBatch ], dim=0 )

    depthOrigin_cpu = dataBatch['depthOrigin']
    depthOriginBatch = depthOrigin_cpu.cuda()
    depthOriginBatch = torch.cat([depthOriginBatch,
        depthOriginBatch, depthOriginBatch ], dim=0 )

    depthScale_cpu = dataBatch['depthScale']
    depthScaleBatch = depthScale_cpu.cuda()
    depthScaleBatch = torch.cat([depthScaleBatch,
        depthScaleBatch, depthScaleBatch ], dim=0 )

    # Load segmentation masks
    segAll_cpu = dataBatch['segAll']
    segObj_cpu = dataBatch['segObj']
    segBRDFBatch = segObj_cpu.cuda()
    segAllBatch = segAll_cpu.cuda()

    segAllDl_cpu = dataBatch['segAllDl']
    segObjDl_cpu = dataBatch['segObjDl']
    segBRDFDlBatch = segObjDl_cpu.cuda()
    segAllDlBatch = segAllDl_cpu.cuda()

    segBRDFBatch = torch.cat([segBRDFBatch,
        segBRDFDlBatch, segBRDFBatch], dim=0 )
    segAllBatch = torch.cat([segAllBatch,
        segAllDlBatch, segAllBatch], dim=0 )

    return albedoBatch, normalBatch, roughBatch, \
            depthBatch, depthOriginBatch, depthScaleBatch, \
            segBRDFBatch, segAllBatch


def getVisWindow(dataBatch ):
    visWinNum = dataBatch['visWinNum']
    visWinNum = visWinNum.numpy().squeeze(1)

    winMasks_cpu = dataBatch['winMasks']
    winMasksBatch = (winMasks_cpu ).cuda()
    winMasksBatch = torch.cat([winMasksBatch,
        winMasksBatch, winMasksBatch], dim=0 )

    onWinMasks_cpu = dataBatch['onWinMasks']
    onWinMasksBatch = (onWinMasks_cpu ).cuda()
    onWinMasksDl_cpu = dataBatch['onWinMasksDl']
    onWinMasksDlBatch = (onWinMasksDl_cpu ).cuda()
    onWinMasksBatch = torch.cat([onWinMasksBatch,
        onWinMasksDlBatch, onWinMasksBatch], dim=0 )

    visWinPlanes_cpu = dataBatch['visWinPlanes']
    visWinPlanesBatch = (visWinPlanes_cpu ).cuda()
    visWinPlanesBatch = torch.cat([visWinPlanesBatch,
        visWinPlanesBatch, visWinPlanesBatch], dim=0 )

    visWinSrcs_cpu = dataBatch['visWinSrcs']
    visWinSrcsBatch = (visWinSrcs_cpu ).cuda()
    visWinSrcsDl_cpu = dataBatch['visWinSrcsDl']
    visWinSrcsDlBatch = (visWinSrcsDl_cpu ).cuda()
    visWinSrcsDm_cpu = dataBatch['visWinSrcsDm']
    visWinSrcsDmBatch = (visWinSrcsDm_cpu ).cuda()
    visWinSrcsBatch = torch.cat([visWinSrcsBatch,
        visWinSrcsDlBatch, visWinSrcsDmBatch ], dim=0 )

    visWinShadingsNo_cpu = dataBatch['visWinShadingsNo']
    visWinShadingsNoBatch = (visWinShadingsNo_cpu ).cuda()
    visWinShadingsNoDl_cpu = dataBatch['visWinShadingsNoDl']
    visWinShadingsNoDlBatch = (visWinShadingsNoDl_cpu ).cuda()
    visWinShadingsNoDm_cpu = dataBatch['visWinShadingsNoDm']
    visWinShadingsNoDmBatch = (visWinShadingsNoDm_cpu ).cuda()
    visWinShadingsNoBatch = torch.cat( [visWinShadingsNoBatch,
        visWinShadingsNoDlBatch, visWinShadingsNoDmBatch ], dim=0 )

    visWinShadingsNoApp_cpu = dataBatch['visWinShadingsNoApp']
    visWinShadingsNoAppBatch = (visWinShadingsNoApp_cpu ).cuda()
    visWinShadingsNoAppDl_cpu = dataBatch['visWinShadingsNoAppDl']
    visWinShadingsNoAppDlBatch = (visWinShadingsNoAppDl_cpu ).cuda()
    visWinShadingsNoAppDm_cpu = dataBatch['visWinShadingsNoAppDm']
    visWinShadingsNoAppDmBatch = (visWinShadingsNoAppDm_cpu ).cuda()
    visWinShadingsNoAppBatch = torch.cat( [visWinShadingsNoAppBatch,
        visWinShadingsNoAppDlBatch, visWinShadingsNoAppDmBatch ], dim=0 )

    visWinShadows_cpu = dataBatch['visWinShadows']
    visWinShadowsBatch = (visWinShadows_cpu ).cuda()
    visWinShadowsBatch = torch.cat( [visWinShadowsBatch,
        visWinShadowsBatch, visWinShadowsBatch ], dim=0 )

    return visWinNum, \
            winMasksBatch, onWinMasksBatch, \
            visWinPlanesBatch, visWinSrcsBatch, \
            visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
            visWinShadowsBatch


class BatchLoader(Dataset):
    def __init__(self, dataRoot,
            imHeight = 240, imWidth = 320,
            phase='TRAIN', rseed = None,
            isLightSrc = False,
            envHeight = 8, envWidth = 16,
            envRow = 120, envCol = 160,
            maxWinNum = 3, maxLampNum = 7, isPredDepth = False ):

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()

        self.isLightSrc = isLightSrc

        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.maxWinNum = maxWinNum
        self.maxLampNum = maxLampNum

        # Load shapes
        shapeList = []
        dirs = ['main_xml', 'main_xml1']
        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        elif phase.upper() == 'DEBUG':
            sceneFile = osp.join(dataRoot, 'debug.txt')
        else:
            print('Wrong: unrecognizable phase')
            assert(False )

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [x.strip() for x in sceneList ]

        for d in dirs:
            shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList]
        shapeList = sorted(shapeList )
        print('Shape Num: %d' % len(shapeList ) )

        self.imList = []
        for shape in shapeList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames

        self.imDlList = [x.replace('main_', 'mainDiffLight_') for x in self.imList ]
        self.imDmList = [x.replace('main_', 'mainDiffMat_') for x in self.imList ]

        imList, imDlList, imDmList = [], [], []
        for n in range(0, len(self.imList ) ):
            imName = self.imList[n]
            imDlName = self.imDlList[n]
            imDmName = self.imDmList[n]

            lightDir = imName.replace('im_', 'light_').replace('.hdr', '')
            lightDlDir = imDlName.replace('im_', 'light_').replace('.hdr', '' )

            visFile = osp.join(lightDir, 'visibility.dat')
            visDlFile = osp.join(lightDir, 'visibility.dat')

            with open(visFile, 'rb') as fIn:
                info = pickle.load(fIn )
            with open(visDlFile, 'rb') as fIn:
                infoDl = pickle.load(fIn )
            if info['isVisWindow'] or infoDl['isVisWindow']:
                imList.append(imName )
                imDlList.append(imDlName )
                imDmList.append(imDmName )

        self.imList = imList
        self.imDlList = imDlList
        self.imDmList = imDmList

        print('Image Num: %d' % (len(self.imList ) * 3) )

        self.semLabelList = [x.replace('im_', 'imsemLabel_').replace('.hdr', '.npy') for x in self.imList ]

        # BRDF parameter
        self.albedoList = [x.replace('im_', 'imbaseColor_').replace('hdr', 'png') for x in self.imList ]
        self.albedoDmList = [x.replace('main_', 'mainDiffMat_') for x in self.albedoList ]

        self.normalList = [x.replace('im_', 'imnormal_').replace('hdr', 'png') for x in self.imList ]
        self.normalDmList = [x.replace('main_', 'mainDiffMat_') for x in self.normalList ]

        self.roughList = [x.replace('im_', 'imroughness_').replace('hdr', 'png') for x in self.imList ]
        self.roughDmList = [x.replace('main_', 'mainDiffMat_') for x in self.roughList ]

        if isPredDepth:
            self.depthList = [x.replace('im_', 'predDepth_').replace('hdr', 'dat') for x in self.imList ]
        else:
            self.depthList = [x.replace('im_', 'imdepth_').replace('hdr', 'dat') for x in self.imList ]

        self.segList = [x.replace('im_', 'immask_').replace('hdr', 'png') for x in self.imList ]
        self.segDlList = [x.replace('main_', 'mainDiffLight_') for x in self.segList ]

        if self.isLightSrc:
            self.lightMaskList = []
            self.lightBoxList = []

            self.lightSrcList = []
            self.lightSrcDlList = []

            self.lightShadingList = []
            self.lightShadingDlList = []

            self.lightShadingNoList = []
            self.lightShadingNoDlList = []

            self.lightShadingNoAppList = []
            self.lightShadingNoAppDlList = []

            self.lightShadowList = []

            for x in self.imList:
                lightDir = x.replace('im_', 'light_').replace('.hdr', '')
                lightMaskNames = glob.glob(osp.join(lightDir, 'mask*.png') )
                lightBoxNames = [x.replace('mask', 'box').replace('.png', '.dat')
                        for x in lightMaskNames ]

                lightShadingNoNames = [x.replace('mask', 'imDSNoOcclu').replace('.png', '.rgbe')
                        for x in  lightMaskNames ]
                lightShadingNoDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightShadingNoNames ]

                lightShadingNoAppNames = [x.replace('mask', 'imDSNoOccluApprox').replace('.png', '.hdr')
                        for x in  lightMaskNames ]
                lightShadingNoAppDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightShadingNoAppNames ]

                lightShadowNames = [x.replace('mask', 'imShadow') for x in lightMaskNames ]

                lightSrcNames = [x.replace('mask', 'lightSrcNoOcclu').replace('.png', '.dat')
                        for x in lightMaskNames ]
                lightSrcDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightSrcNames ]


                self.lightMaskList.append(lightMaskNames )
                self.lightBoxList.append(lightBoxNames )

                self.lightShadingNoList.append(lightShadingNoNames )
                self.lightShadingNoDlList.append(lightShadingNoDlNames )

                self.lightShadingNoAppList.append(lightShadingNoAppNames )
                self.lightShadingNoAppDlList.append(lightShadingNoAppDlNames )

                self.lightSrcList.append(lightSrcNames )
                self.lightSrcDlList.append(lightSrcDlNames )

                self.lightShadowList.append(lightShadowNames )


        # Permute the image list
        self.count = len(self.imList )
        self.perm = list(range(self.count ) )

        if rseed is not None:
            random.seed(rseed )

        if self.phase.upper() == 'TRAIN':
            random.shuffle(self.perm )

        np.random.seed(int(time.time() ) )

    def __len__(self):
        return len(self.perm )

    def __getitem__(self, ind):
        # Read segmentation
        segObj, segEnv, segArea = self.loadSeg(self.segList[self.perm[ind ] ] )
        segObjDl, segEnvDl, segAreaDl = self.loadSeg(self.segDlList[self.perm[ind] ]  )

        segAll = np.clip(1 - segEnv, 0, 1 )
        segAllDl = np.clip(1 - segEnvDl, 0, 1 )

        # Read Image
        im = self.loadHdr(self.imList[self.perm[ind ] ] )
        im, scale = self.scaleHdr(im, segObj )
        imDl = self.loadHdr(self.imDlList[self.perm[ind ] ] )
        imDl, scaleDl = self.scaleHdr(imDl, segObjDl )
        imDm = self.loadHdr(self.imDmList[self.perm[ind] ] )
        imDm, scaleDm = self.scaleHdr(imDm, segObj )

        # Read label and compute window mask
        semLabel = np.load(self.semLabelList[self.perm[ind ] ] )
        semLabel = (1 - (semLabel == 31) ).astype(np.float32 )
        semLabel = cv2.resize(semLabel, (self.imWidth, self.imHeight ), interpolation = cv2.INTER_AREA )
        semLabel = ndimage.binary_erosion(semLabel == 1, structure = np.ones( (5, 5) ) )
        semLabel = semLabel.astype(np.float32 )[np.newaxis, :]

        # Read albedo
        albedo = self.loadImage(self.albedoList[self.perm[ind] ], isGama = False)
        albedo = (0.5 * (albedo + 1) ) ** 2.2
        albedoDm = self.loadImage(self.albedoDmList[self.perm[ind] ], isGama = False)
        albedoDm = (0.5 * (albedoDm + 1) ) ** 2.2

        # normalize the normal vector so that it will be unit length
        normal = self.loadImage(self.normalList[self.perm[ind] ] )
        normal = normal / np.sqrt(np.maximum(np.sum(normal * normal, axis=0), \
                1e-5) )[np.newaxis, :]
        normalDm = self.loadImage(self.normalDmList[self.perm[ind] ] )
        normalDm = normalDm / np.sqrt(np.maximum(np.sum(normalDm * normalDm, axis=0), \
                1e-5) )[np.newaxis, :]

        # Read roughness
        rough = self.loadImage(self.roughList[self.perm[ind] ] )[0:1, :, :]
        rough = 0.5 * (rough + 1)
        roughDm = self.loadImage(self.roughDmList[self.perm[ind] ] )[0:1, :, :]
        roughDm = 0.5 * (roughDm + 1)

        # Read depth
        depthOrigin = self.loadBinary(self.depthList[self.perm[ind] ] )
        depthScale = np.maximum(np.mean(depthOrigin ), 1) / 3.0
        depth = depthOrigin / depthScale

        if self.isLightSrc == True:
            lightMaskNames = self.lightMaskList[self.perm[ind ] ]
            lightBoxNames = self.lightBoxList[self.perm[ind ] ]

            shadingNoNames = self.lightShadingNoList[self.perm[ind ] ]
            shadingNoDlNames = self.lightShadingNoDlList[self.perm[ind ] ]

            shadingNoAppNames = self.lightShadingNoAppList[self.perm[ind ] ]
            shadingNoAppDlNames = self.lightShadingNoAppDlList[self.perm[ind ] ]

            shadowNames = self.lightShadowList[self.perm[ind ] ]

            lightSrcNames = self.lightSrcList[self.perm[ind ] ]
            lightSrcDlNames = self.lightSrcDlList[self.perm[ind ] ]

            # Visible lamps
            visLampMask = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth ), dtype = np.float32 )
            onLampMask = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth), dtype = np.float32 )
            onLampMaskDl = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth), dtype = np.float32 )

            visLampNum = np.zeros( (1 ), dtype = np.int32 )

            # Visible window Light
            visWinMask = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth ), dtype = np.float32 )
            onWinMask = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth), dtype = np.float32 )
            onWinMaskDl = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth), dtype = np.float32 )

            visWinPlane = np.zeros( (self.maxWinNum, 12 ), dtype = np.float32 )

            visWinSrc = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            visWinSrcDl = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            visWinSrcDm = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )

            visWinShadingNo = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visWinShadingNoApp = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoAppDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoAppDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )

            visWinNum = np.zeros( (1 ), dtype = np.int32 )

            # raw parameters for light source
            for n in range(0, len(lightMaskNames ) ):
                maskName = lightMaskNames[n ]
                boxName = lightBoxNames[n ]

                shNoName = shadingNoNames[n ]
                shNoDlName = shadingNoDlNames[n ]

                shNoAppName = shadingNoAppNames[n ]
                shNoAppDlName = shadingNoAppDlNames[n ]

                shdName = shadowNames[n ]

                srcName = lightSrcNames[n ]
                srcDlName = lightSrcDlNames[n ]

                mask = cv2.imread(maskName )[:, :, 0]
                mask = cv2.resize(mask, (self.imWidth, self.imHeight),
                        interpolation = cv2.INTER_AREA )
                mask = mask.astype(np.float32 ) / 255.0
                with open(boxName, 'rb') as fIn:
                    boxPara = pickle.load(fIn )
                with open(srcName, 'rb') as fIn:
                    srcPara = pickle.load(fIn )
                with open(srcDlName, 'rb') as fIn:
                    srcDlPara = pickle.load(fIn )

                lightShadingNo = self.loadHdr(shNoName, self.envRow, self.envCol )
                lightShadingNoDl = self.loadHdr(shNoDlName, self.envRow, self.envCol )

                lightShadow = cv2.imread(shdName )[:, :, 0]
                if lightShadow.shape[0] != self.envRow:
                    lightShadow = cv2.resize(lightShadow, (self.envRow, self.envCol ),
                                             interpolation = cv2.INTER_AREA )
                lightShadow = (lightShadow.astype(np.float32 ) / 255.0)[np.newaxis, :]

                intensity = srcPara['intensity'] * scale
                intensityDl = srcDlPara['intensity'] * scaleDl
                intensityDm = srcPara['intensity'] * scaleDm

                box3DPara = boxPara['box3D']
                center = box3DPara['center'].reshape(3) / depthScale
                xAxis = (box3DPara['xAxis'] * box3DPara['xLen'] ).reshape(3) / depthScale
                yAxis = (box3DPara['yAxis'] * box3DPara['yLen'] ).reshape(3) / depthScale
                zAxisNormalized = -box3DPara['zAxis'].reshape(3)
                zAxis = (-box3DPara['zAxis'] * box3DPara['zLen'] ).reshape(3) / depthScale

                if np.sum(center * zAxisNormalized ) < 0:
                    zAxisNormalized = - zAxisNormalized
                    zAxis = -zAxis


                if boxPara['isWindow']:
                    winSrc = np.concatenate([
                        srcPara['intensity'] * scale, srcPara['axis'], np.array(srcPara['lamb'] ).reshape(1),
                        srcPara['intensitySky'] * scale, srcPara['axisSky'], np.array(srcPara['lambSky'] ).reshape(1),
                        srcPara['intensityGrd'] * scale, srcPara['axisGrd'], np.array(srcPara['lambGrd'] ).reshape(1),
                        np.zeros(1, dtype= np.float32 )
                    ], axis=0 )
                    winSrcDl = np.concatenate([
                        srcDlPara['intensity'] * scaleDl, srcDlPara['axis'], np.array(srcDlPara['lamb'] ).reshape(1),
                        srcDlPara['intensitySky'] * scaleDl, srcDlPara['axisSky'], np.array(srcDlPara['lambSky'] ).reshape(1),
                        srcDlPara['intensityGrd'] * scaleDl, srcDlPara['axisGrd'], np.array(srcDlPara['lambGrd'] ).reshape(1),
                        np.zeros(1, dtype = np.float32 )
                    ], axis=0 )
                    winSrcDm = np.concatenate([
                        srcPara['intensity'] * scaleDm, srcPara['axis'], np.array(srcPara['lamb'] ).reshape(1),
                        srcPara['intensitySky'] * scaleDm, srcPara['axisSky'], np.array(srcPara['lambSky'] ).reshape(1),
                        srcPara['intensityGrd'] * scaleDm, srcPara['axisGrd'], np.array(srcPara['lambGrd'] ).reshape(1),
                        np.zeros(1, dtype = np.float32 )
                    ], axis=0 )

                if np.sum(mask ) > 25:
                    # Light Masks
                    if boxPara['isWindow']:
                        visWinMask[visWinNum[0], :] = mask
                        if np.sum(intensity ) > 0:
                            onWinMask[visWinNum[0], :] = mask

                            lightShadingNoApp = self.loadHdr(shNoAppName, self.envRow, self.envCol ) * scale
                            lightShadingNoAppDm = lightShadingNoApp / scale * scaleDm

                            visWinShadingNoApp[visWinNum[0], ] = lightShadingNoApp
                            visWinShadingNoAppDm[visWinNum[0], ] = lightShadingNoAppDm

                            winSrc[-1]  = srcPara['loss'] / srcPara['scale'] * scale
                            winSrcDm[-1] = srcPara['loss'] / srcPara['scale'] * scaleDm

                        if np.sum(intensityDl ) > 0:
                            onWinMaskDl[visWinNum[0], :] = mask

                            lightShadingNoAppDl = self.loadHdr(shNoAppDlName, self.envRow, self.envCol ) * scaleDl
                            visWinShadingNoAppDl[visWinNum[0], :] = lightShadingNoAppDl

                            winSrcDl[-1] = srcDlPara['loss'] / srcDlPara['scale'] * scaleDl

                        visWinSrc[visWinNum[0], :] = winSrc
                        visWinSrcDl[visWinNum[0], :] = winSrcDl
                        visWinSrcDm[visWinNum[0], :] = winSrcDm

                        # Get the center and the points
                        winCenter = center
                        visWinPlane[visWinNum[0], :] = np.concatenate(
                                [winCenter, zAxisNormalized, yAxis, xAxis], axis=0 )

                        visWinShadingNo[visWinNum[0], ] = lightShadingNo * scale
                        visWinShadingNoDl[visWinNum[0], ] = lightShadingNoDl * scaleDl
                        visWinShadingNoDm[visWinNum[0], ] = lightShadingNo * scaleDm

                        visWinShadow[visWinNum[0], ] = lightShadow

                        visWinNum[0] = visWinNum[0] + 1

                    else:
                        visLampMask[visLampNum[0], :] = mask
                        if np.sum(intensity ) > 0:
                            onLampMask[visLampNum[0], :] = mask
                        if np.sum(intensityDl ) > 0:
                            onLampMaskDl[visLampNum[0], :] = mask

                        visLampNum[0] += 1

        isFlip = (np.random.random() > 0.5) and self.phase.upper() == 'TRAIN'
        if isFlip:
            albedo = np.ascontiguousarray(albedo[:, :, ::-1] )
            albedoDm = np.ascontiguousarray(albedoDm[:, :, ::-1] )

            normal = np.ascontiguousarray(normal[:, :, ::-1] )
            normal[0, :, :] = -normal[0, :, :]

            normalDm = np.ascontiguousarray(normalDm[:, :, ::-1] )
            normalDm[0, :, :] = -normalDm[0, :, :]

            rough = np.ascontiguousarray(rough[:, :, ::-1] )
            roughDm = np.ascontiguousarray(roughDm[:, :, ::-1] )

            depth = np.ascontiguousarray(depth[:, :, ::-1] )
            depthOrigin = np.ascontiguousarray(depthOrigin[:, :, ::-1] )

            segAll = np.ascontiguousarray(segAll[:, :, ::-1] )
            segObj = np.ascontiguousarray(segObj[:, :, ::-1] )
            segAllDl = np.ascontiguousarray(segAllDl[:, :, ::-1] )
            segObjDl = np.ascontiguousarray(segObjDl[:, :, ::-1] )

            im = np.ascontiguousarray(im[:, :, ::-1] )
            imDl = np.ascontiguousarray(imDl[:, :, ::-1] )
            imDm = np.ascontiguousarray(imDm[:, :, ::-1] )

            semLabel = np.ascontiguousarray(semLabel[:, :, ::-1] )

        batchDict = {
                # diffuse albedo
                'albedo': albedo,
                'albedoDm': albedoDm,
                # normal
                'normal': normal,
                'normalDm': normalDm,
                # roughness
                'rough': rough,
                'roughDm': roughDm,
                # depth
                'depth': depth,
                'depthOrigin':  depthOrigin,
                'depthScale': depthScale,
                # segmentation mask
                'segAll': segAll,
                'segObj': segObj,
                'segAllDl': segAllDl,
                'segObjDl': segObjDl,
                # image
                'im': im,
                'imDl': imDl,
                'imDm': imDm,
                # mask
                'semLabel': semLabel,
                # scale
                'scale': scale,
                'scaleDl': scaleDl,
                'scaleDm': scaleDm,
                # name
                'name': self.imList[self.perm[ind ] ]
                }

        if self.isLightSrc:
            lightMask = np.sum(visLampMask, axis=0, keepdims=True ) + np.sum(visWinMask, axis=0, keepdims=True )
            lightMask = np.clip(lightMask, 0, 1 )
            lightOnMask = np.sum(onLampMask, axis=0, keepdims=True ) + np.sum(onWinMask, axis=0, keepdims=True )
            lightOnMask = np.clip(lightOnMask, 0, 1 )
            lightOnMaskDl = np.sum(onLampMaskDl, axis=0, keepdims=True) + np.sum(onWinMaskDl, axis=0, keepdims=True )
            lightOnMaskDl = np.clip(lightOnMaskDl, 0, 1 )

            if isFlip:
                lightMask = np.ascontiguousarray(lightMask[:, :, ::-1] )
                lightOnMask = np.ascontiguousarray(lightOnMask[:, :, ::-1] )
                lightOnMaskDl = np.ascontiguousarray(lightOnMaskDl[:, :, ::-1] )

                visWinMask = np.ascontiguousarray(visWinMask[:, :, ::-1] )
                onWinMask = np.ascontiguousarray(onWinMask[:, :, ::-1] )
                onWinMaskDl = np.ascontiguousarray(onWinMaskDl[:, :, ::-1] )

                visWinPlane[:, 0] = -visWinPlane[:, 0]
                visWinPlane[:, 3] = -visWinPlane[:, 3]
                visWinPlane[:, 6] = -visWinPlane[:, 6]
                visWinPlane[:, 9] = -visWinPlane[:, 9]

                visWinSrc[:, 3] = -visWinSrc[:, 3]
                visWinSrc[:, 10] = -visWinSrc[:, 10]
                visWinSrc[:, 17] = -visWinSrc[:, 17]

                visWinSrcDl[:, 3] =  -visWinSrcDl[:, 3]
                visWinSrcDl[:, 10] =  -visWinSrcDl[:, 10]
                visWinSrcDl[:, 17] =  -visWinSrcDl[:, 17]

                visWinSrcDm[:, 3] = -visWinSrcDm[:, 3]
                visWinSrcDm[:, 10] = -visWinSrcDm[:, 10]
                visWinSrcDm[:, 17] = -visWinSrcDm[:, 17]

                visWinShadingNo = np.ascontiguousarray(visWinShadingNo[:, :, :, ::-1] )
                visWinShadingNoDl = np.ascontiguousarray(visWinShadingNoDl[:, :, :, ::-1] )
                visWinShadingNoDm = np.ascontiguousarray(visWinShadingNoDm[:, :, :, ::-1] )

                visWinShadingNoApp = np.ascontiguousarray(visWinShadingNoApp[:, :, :, ::-1] )
                visWinShadingNoAppDl = np.ascontiguousarray(visWinShadingNoAppDl[:, :, :, ::-1] )
                visWinShadingNoAppDm = np.ascontiguousarray(visWinShadingNoAppDm[:, :, :, ::-1] )

                visWinShadow = np.ascontiguousarray(visWinShadow[:, :, :, ::-1] )

            batchDict['lightMasks'] = lightMask
            batchDict['lightOnMasks'] = lightOnMask
            batchDict['lightOnMasksDl'] = lightOnMaskDl

            # Visible light window
            batchDict['winMasks'] = visWinMask
            batchDict['onWinMasks'] = onWinMask
            batchDict['onWinMasksDl'] = onWinMaskDl

            batchDict['visWinPlanes'] = visWinPlane

            batchDict['visWinSrcs'] = visWinSrc
            batchDict['visWinSrcsDl'] = visWinSrcDl
            batchDict['visWinSrcsDm'] = visWinSrcDm

            batchDict['visWinShadingsNo'] = visWinShadingNo
            batchDict['visWinShadingsNoDl'] = visWinShadingNoDl
            batchDict['visWinShadingsNoDm'] = visWinShadingNoDm

            batchDict['visWinShadingsNoApp'] = visWinShadingNoApp
            batchDict['visWinShadingsNoAppDl'] = visWinShadingNoAppDl
            batchDict['visWinShadingsNoAppDm'] = visWinShadingNoAppDm

            batchDict['visWinShadows'] = visWinShadow

            batchDict['visWinNum'] = visWinNum

        return batchDict


    def loadImage(self, imName, isGama = False):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )

        im = cv2.imread(imName )
        im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation =
                cv2.INTER_AREA )
        im = np.ascontiguousarray(im[:, :, ::-1] )

        im = np.asarray(im, dtype=np.float32 )
        if isGama:
            im = (im / 255.0) ** 2.2
            im = 2 * im - 1
        else:
            im = (im - 127.5) / 127.5
        if len(im.shape) == 2:
            im = im[:, np.newaxis]
        im = np.transpose(im, [2, 0, 1] )

        return im

    def loadSeg(self, name ):
        seg = 0.5 * (self.loadImage(name ) + 1)[0:1, :, :]
        segArea = np.logical_and(seg > 0.05, seg < 0.95 ).astype(np.float32 )

        segEnv = (seg < 0.05 )
        segEnv = segEnv.squeeze()
        segEnv = ndimage.binary_dilation(segEnv, structure=np.ones((5, 5)) )
        segEnv = segEnv[np.newaxis, :, :]
        segEnv = segEnv.astype(np.float32 )


        segObj = (seg > 0.99 )
        segObj = segObj.squeeze()
        segObj = ndimage.binary_erosion(segObj, structure=np.ones( (5, 5) ),
                border_value=1 )
        segObj = segObj[np.newaxis, :, :]
        segObj = segObj.astype(np.float32 )

        return segObj, segEnv, segArea


    def loadHdr(self, imName, imHeight = None, imWidth = None):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )
        im = cv2.imread(imName, -1)
        if im is None:
            print(imName )
            assert(False )

        if imHeight is None or imWidth is None:
            im = cv2.resize(im, (self.imWidth, self.imHeight), interpolation = cv2.INTER_AREA )
        else:
            if imHeight != im.shape[0] or imWidth != im.shape[1]:
                im = cv2.resize(im, (imWidth, imHeight), interpolation = cv2.INTER_AREA )
        im = np.transpose(im, [2, 0, 1] )
        im = im[::-1, :, :]
        im = np.ascontiguousarray(im )
        return im

    def scaleHdr(self, hdr, seg):
        intensityArr = (hdr * seg).flatten()
        intensityArr.sort()
        if self.phase.upper() == 'TRAIN':
            scale = (1.0 - 0.1 * np.random.random() )  \
                    / np.clip(intensityArr[int(0.98*self.imWidth*self.imHeight*3) ], 0.1, None )
        else:
            scale = (1.0 - 0.05 )  \
                    / np.clip(intensityArr[int(0.98*self.imWidth*self.imHeight*3) ], 0.1, None )
        hdr = scale * hdr
        return np.clip(hdr, 0, 1), scale

    def loadBinary(self, imName ):
        if not(osp.isfile(imName ) ):
            print(imName )
            assert(False )
        with open(imName, 'rb') as fIn:
            hBuffer = fIn.read(4)
            height = struct.unpack('i', hBuffer)[0]
            wBuffer = fIn.read(4)
            width = struct.unpack('i', wBuffer)[0]
            dBuffer = fIn.read(4 * width * height )
            depth = np.asarray(struct.unpack('f' * height * width, dBuffer), dtype=np.float32 )
            depth = depth.reshape([height, width] )
            depth = cv2.resize(depth, (self.imWidth, self.imHeight), interpolation=cv2.INTER_AREA )
        return depth[np.newaxis, :, :]
