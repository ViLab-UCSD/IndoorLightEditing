import glob
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


def getVisWindow(dataBatch, isDepthShadow = False ):
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

    visWinShadings_cpu = dataBatch['visWinShadings']
    visWinShadingsBatch = (visWinShadings_cpu ).cuda()
    visWinShadingsDl_cpu = dataBatch['visWinShadingsDl']
    visWinShadingsDlBatch = (visWinShadingsDl_cpu ).cuda()
    visWinShadingsDm_cpu = dataBatch['visWinShadingsDm']
    visWinShadingsDmBatch = (visWinShadingsDm_cpu ).cuda()
    visWinShadingsBatch = torch.cat( [visWinShadingsBatch,
        visWinShadingsDlBatch, visWinShadingsDmBatch ], dim=0 )

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

    if isDepthShadow:
        depthVisWinShadows_cpu = dataBatch['depthVisWinShadows']
        depthVisWinShadowsBatch = (depthVisWinShadows_cpu ).cuda()
        depthVisWinShadowsBatch = torch.cat( [depthVisWinShadowsBatch,
            depthVisWinShadowsBatch, depthVisWinShadowsBatch ], dim=0 )

        confVisWinShadows_cpu = dataBatch['confVisWinShadows']
        confVisWinShadowsBatch = (confVisWinShadows_cpu ).cuda()
        confVisWinShadowsBatch = torch.cat( [confVisWinShadowsBatch,
            confVisWinShadowsBatch, confVisWinShadowsBatch ], dim=0 )

        return visWinNum, \
                winMasksBatch, onWinMasksBatch, \
                visWinPlanesBatch, visWinSrcsBatch, \
                visWinShadingsBatch, visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
                visWinShadowsBatch,\
                depthVisWinShadowsBatch, confVisWinShadowsBatch
    else:
        return visWinNum, \
                winMasksBatch, onWinMasksBatch, \
                visWinPlanesBatch, visWinSrcsBatch, \
                visWinShadingsBatch, visWinShadingsNoBatch, visWinShadingsNoAppBatch, \
                visWinShadowsBatch

def getVisLamp(dataBatch, isDepthShadow = False ):
    visLampNum = dataBatch['visLampNum']
    visLampNum = visLampNum.numpy().squeeze(1)

    lampMasks_cpu = dataBatch['lampMasks']
    lampMasksBatch = (lampMasks_cpu ).cuda()
    lampMasksBatch = torch.cat([lampMasksBatch,
        lampMasksBatch, lampMasksBatch], dim=0 )

    onLampMasks_cpu = dataBatch['onLampMasks']
    onLampMasksBatch = (onLampMasks_cpu ).cuda()
    onLampMasksDl_cpu = dataBatch['onLampMasksDl']
    onLampMasksDlBatch = (onLampMasksDl_cpu ).cuda()
    onLampMasksBatch = torch.cat([onLampMasksBatch,
        onLampMasksDlBatch, onLampMasksBatch], dim=0 )

    visLampCenters_cpu = dataBatch['visLampCenters']
    visLampCentersBatch = (visLampCenters_cpu ).cuda()
    visLampCentersBatch = torch.cat([visLampCentersBatch,
        visLampCentersBatch, visLampCentersBatch], dim=0 )

    visLampAxes_cpu = dataBatch['visLampAxes']
    visLampAxesBatch = (visLampAxes_cpu ).cuda()
    visLampAxesBatch = torch.cat([visLampAxesBatch,
        visLampAxesBatch, visLampAxesBatch], dim=0 )

    visLampShadings_cpu = dataBatch['visLampShadings']
    visLampShadingsBatch = (visLampShadings_cpu ).cuda()
    visLampShadingsDl_cpu = dataBatch['visLampShadingsDl']
    visLampShadingsDlBatch = (visLampShadingsDl_cpu ).cuda()
    visLampShadingsDm_cpu = dataBatch['visLampShadingsDm']
    visLampShadingsDmBatch = (visLampShadingsDm_cpu ).cuda()
    visLampShadingsBatch = torch.cat( [visLampShadingsBatch,
        visLampShadingsDlBatch, visLampShadingsDmBatch ], dim=0 )

    visLampShadingsNo_cpu = dataBatch['visLampShadingsNo']
    visLampShadingsNoBatch = (visLampShadingsNo_cpu ).cuda()
    visLampShadingsNoDl_cpu = dataBatch['visLampShadingsNoDl']
    visLampShadingsNoDlBatch = (visLampShadingsNoDl_cpu ).cuda()
    visLampShadingsNoDm_cpu = dataBatch['visLampShadingsNoDm']
    visLampShadingsNoDmBatch = (visLampShadingsNoDm_cpu ).cuda()
    visLampShadingsNoBatch = torch.cat( [visLampShadingsNoBatch,
        visLampShadingsNoDlBatch, visLampShadingsNoDmBatch ], dim=0 )

    visLampShadows_cpu = dataBatch['visLampShadows']
    visLampShadowsBatch = (visLampShadows_cpu ).cuda()
    visLampShadowsBatch = torch.cat( [visLampShadowsBatch,
        visLampShadowsBatch, visLampShadowsBatch ], dim=0 )

    if isDepthShadow == True:
        depthVisLampShadows_cpu = dataBatch['depthVisLampShadows']
        depthVisLampShadowsBatch = (depthVisLampShadows_cpu ).cuda()
        depthVisLampShadowsBatch = torch.cat( [depthVisLampShadowsBatch,
            depthVisLampShadowsBatch, depthVisLampShadowsBatch ], dim=0 )

        confVisLampShadows_cpu = dataBatch['confVisLampShadows']
        confVisLampShadowsBatch = (confVisLampShadows_cpu ).cuda()
        confVisLampShadowsBatch = torch.cat( [confVisLampShadowsBatch,
            confVisLampShadowsBatch, confVisLampShadowsBatch ], dim=0 )

        return  visLampNum, lampMasksBatch, onLampMasksBatch, \
                visLampCentersBatch, visLampAxesBatch, \
                visLampShadingsBatch, visLampShadingsNoBatch, \
                visLampShadowsBatch, depthVisLampShadowsBatch,  \
                confVisLampShadowsBatch
    else:
        return  visLampNum, lampMasksBatch, onLampMasksBatch, \
                visLampCentersBatch, visLampAxesBatch, \
                visLampShadingsBatch, visLampShadingsNoBatch, \
                visLampShadowsBatch

def getInvWindow(dataBatch, isDepthShadow = False ):
    invWinNum = dataBatch['invWinNum']
    invWinNum = invWinNum.numpy().squeeze(1)

    invWinOn_cpu = dataBatch['invWinOn']
    invWinOnBatch = (invWinOn_cpu ).cuda()
    invWinOnDl_cpu = dataBatch['invWinOnDl']
    invWinOnDlBatch = (invWinOnDl_cpu ).cuda()
    invWinOnBatch = torch.cat([invWinOnBatch,
        invWinOnDlBatch, invWinOnBatch], dim=0 )

    invWinPlanes_cpu = dataBatch['invWinPlanes']
    invWinPlanesBatch = (invWinPlanes_cpu ).cuda()
    invWinPlanesBatch = torch.cat([invWinPlanesBatch,
        invWinPlanesBatch, invWinPlanesBatch], dim=0 )

    invWinSrcs_cpu = dataBatch['invWinSrcs']
    invWinSrcsBatch = invWinSrcs_cpu.cuda()
    invWinSrcsDl_cpu = dataBatch['invWinSrcsDl']
    invWinSrcsDlBatch = invWinSrcsDl_cpu.cuda()
    invWinSrcsDm_cpu = dataBatch['invWinSrcsDm']
    invWinSrcsDmBatch = invWinSrcsDm_cpu.cuda()
    invWinSrcsBatch = torch.cat([invWinSrcsBatch,
        invWinSrcsDlBatch, invWinSrcsDmBatch ], dim=0 )

    invWinShadings_cpu = dataBatch['invWinShadings']
    invWinShadingsBatch = (invWinShadings_cpu ).cuda()
    invWinShadingsDl_cpu = dataBatch['invWinShadingsDl']
    invWinShadingsDlBatch = (invWinShadingsDl_cpu ).cuda()
    invWinShadingsDm_cpu = dataBatch['invWinShadingsDm']
    invWinShadingsDmBatch = (invWinShadingsDm_cpu ).cuda()
    invWinShadingsBatch = torch.cat( [invWinShadingsBatch,
        invWinShadingsDlBatch, invWinShadingsDmBatch ], dim=0 )

    invWinShadingsNo_cpu = dataBatch['invWinShadingsNo']
    invWinShadingsNoBatch = (invWinShadingsNo_cpu ).cuda()
    invWinShadingsNoDl_cpu = dataBatch['invWinShadingsNoDl']
    invWinShadingsNoDlBatch = (invWinShadingsNoDl_cpu ).cuda()
    invWinShadingsNoDm_cpu = dataBatch['invWinShadingsNoDm']
    invWinShadingsNoDmBatch = (invWinShadingsNoDm_cpu ).cuda()
    invWinShadingsNoBatch = torch.cat( [invWinShadingsNoBatch,
        invWinShadingsNoDlBatch, invWinShadingsNoDmBatch ], dim=0 )

    invWinShadingsNoApp_cpu = dataBatch['invWinShadingsNoApp']
    invWinShadingsNoAppBatch = (invWinShadingsNoApp_cpu ).cuda()
    invWinShadingsNoAppDl_cpu = dataBatch['invWinShadingsNoAppDl']
    invWinShadingsNoAppDlBatch = invWinShadingsNoAppDl_cpu.cuda()
    invWinShadingsNoAppDm_cpu = dataBatch['invWinShadingsNoAppDm']
    invWinShadingsNoAppDmBatch = invWinShadingsNoAppDm_cpu.cuda()
    invWinShadingsNoAppBatch = torch.cat( [invWinShadingsNoAppBatch,
        invWinShadingsNoAppDlBatch, invWinShadingsNoAppDmBatch ], dim=0 )

    invWinShadows_cpu = dataBatch['invWinShadows']
    invWinShadowsBatch = (invWinShadows_cpu ).cuda()
    invWinShadowsBatch = torch.cat( [invWinShadowsBatch,
        invWinShadowsBatch, invWinShadowsBatch ], dim=0 )

    if isDepthShadow == True:
        depthInvWinShadows_cpu = dataBatch['depthInvWinShadows']
        depthInvWinShadowsBatch = (depthInvWinShadows_cpu ).cuda()
        depthInvWinShadowsBatch = torch.cat( [depthInvWinShadowsBatch,
            depthInvWinShadowsBatch, depthInvWinShadowsBatch ], dim=0 )

        confInvWinShadows_cpu = dataBatch['confInvWinShadows']
        confInvWinShadowsBatch = (confInvWinShadows_cpu ).cuda()
        confInvWinShadowsBatch = torch.cat( [confInvWinShadowsBatch,
            confInvWinShadowsBatch, confInvWinShadowsBatch ], dim=0 )

        return invWinNum, \
                invWinOnBatch, \
                invWinPlanesBatch, invWinSrcsBatch, \
                invWinShadingsBatch, invWinShadingsNoBatch, invWinShadingsNoAppBatch, \
                invWinShadowsBatch, \
                depthInvWinShadowsBatch, confInvWinShadowsBatch
    else:
        return invWinNum, \
                invWinOnBatch, \
                invWinPlanesBatch, invWinSrcsBatch, \
                invWinShadingsBatch, invWinShadingsNoBatch, invWinShadingsNoAppBatch, \
                invWinShadowsBatch

def getInvLamp(dataBatch, isDepthShadow = False ):
    invLampNum = dataBatch['invLampNum']
    invLampNum = invLampNum.numpy().squeeze(1)

    invLampOn_cpu = dataBatch['invLampOn']
    invLampOnBatch = (invLampOn_cpu ).cuda()
    invLampOnDl_cpu = dataBatch['invLampOnDl']
    invLampOnDlBatch = (invLampOnDl_cpu ).cuda()
    invLampOnBatch = torch.cat([invLampOnBatch,
        invLampOnDlBatch, invLampOnBatch], dim=0 )

    invLampCenters_cpu = dataBatch['invLampCenters']
    invLampCentersBatch = (invLampCenters_cpu ).cuda()
    invLampCentersBatch = torch.cat([invLampCentersBatch,
        invLampCentersBatch, invLampCentersBatch], dim=0 )

    invLampAxes_cpu = dataBatch['invLampAxes']
    invLampAxesBatch = (invLampAxes_cpu ).cuda()
    invLampAxesBatch = torch.cat([invLampAxesBatch,
        invLampAxesBatch, invLampAxesBatch], dim=0 )

    invLampShadings_cpu = dataBatch['invLampShadings']
    invLampShadingsBatch = (invLampShadings_cpu ).cuda()
    invLampShadingsDl_cpu = dataBatch['invLampShadingsDl']
    invLampShadingsDlBatch = (invLampShadingsDl_cpu ).cuda()
    invLampShadingsDm_cpu = dataBatch['invLampShadingsDm']
    invLampShadingsDmBatch = (invLampShadingsDm_cpu ).cuda()
    invLampShadingsBatch = torch.cat( [invLampShadingsBatch,
        invLampShadingsDlBatch, invLampShadingsDmBatch ], dim=0 )

    invLampShadingsNo_cpu = dataBatch['invLampShadingsNo']
    invLampShadingsNoBatch = (invLampShadingsNo_cpu ).cuda()
    invLampShadingsNoDl_cpu = dataBatch['invLampShadingsNoDl']
    invLampShadingsNoDlBatch = (invLampShadingsNoDl_cpu ).cuda()
    invLampShadingsNoDm_cpu = dataBatch['invLampShadingsNoDm']
    invLampShadingsNoDmBatch = (invLampShadingsNoDm_cpu ).cuda()
    invLampShadingsNoBatch = torch.cat( [invLampShadingsNoBatch,
        invLampShadingsNoDlBatch, invLampShadingsNoDmBatch ], dim=0 )

    invLampShadows_cpu = dataBatch['invLampShadows']
    invLampShadowsBatch = (invLampShadows_cpu ).cuda()
    invLampShadowsBatch = torch.cat( [invLampShadowsBatch,
        invLampShadowsBatch, invLampShadowsBatch ], dim=0 )

    if isDepthShadow == True:
        depthInvLampShadows_cpu = dataBatch['depthInvLampShadows']
        depthInvLampShadowsBatch = (depthInvLampShadows_cpu ).cuda()
        depthInvLampShadowsBatch = torch.cat( [depthInvLampShadowsBatch,
            depthInvLampShadowsBatch, depthInvLampShadowsBatch ], dim=0 )

        confInvLampShadows_cpu = dataBatch['confInvLampShadows']
        confInvLampShadowsBatch = (confInvLampShadows_cpu ).cuda()
        confInvLampShadowsBatch = torch.cat( [confInvLampShadowsBatch,
            confInvLampShadowsBatch, confInvLampShadowsBatch ], dim=0 )

        return invLampNum, \
                invLampOnBatch, \
                invLampCentersBatch, invLampAxesBatch, \
                invLampShadingsBatch, invLampShadingsNoBatch, \
                invLampShadowsBatch, \
                depthInvLampShadowsBatch, confInvLampShadowsBatch
    else:
        return invLampNum, \
                invLampOnBatch, \
                invLampCentersBatch, invLampAxesBatch, \
                invLampShadingsBatch, invLampShadingsNoBatch, \
                invLampShadowsBatch


def getShading(dataBatch ):
    shading_cpu = dataBatch['shading']
    shadingBatch = (shading_cpu ).cuda()
    shadingDl_cpu = dataBatch['shadingDl']
    shadingDlBatch = (shadingDl_cpu ).cuda()
    shadingDm_cpu = dataBatch['shadingDm']
    shadingDmBatch = (shadingDm_cpu ).cuda()
    shadingBatch = torch.cat([shadingBatch, \
            shadingDlBatch, shadingDmBatch ], dim=0 )

    shadingDirect_cpu = dataBatch['shadingDirect']
    shadingDirectBatch = (shadingDirect_cpu ).cuda()
    shadingDirectDl_cpu = dataBatch['shadingDirectDl']
    shadingDirectDlBatch = (shadingDirectDl_cpu ).cuda()
    shadingDirectDm_cpu = dataBatch['shadingDirectDm']
    shadingDirectDmBatch = (shadingDirectDm_cpu ).cuda()
    shadingDirectBatch = torch.cat([shadingDirectBatch, \
            shadingDirectDlBatch, shadingDirectDmBatch ], dim=0 )

    return shadingBatch, shadingDirectBatch

def getEnvmap(dataBatch ):
    envmap_cpu = dataBatch['envmap']
    envmapBatch = (envmap_cpu ).cuda()
    envmapDl_cpu = dataBatch['envmapDl']
    envmapDlBatch = (envmapDl_cpu ).cuda()
    envmapDm_cpu = dataBatch['envmapDm']
    envmapDmBatch = (envmapDm_cpu ).cuda()
    envmapBatch = torch.cat([envmapBatch, \
            envmapDlBatch, envmapDmBatch ], dim=0 )

    envmapInd_cpu = dataBatch['envmapInd']
    envmapIndBatch = (envmapInd_cpu ).cuda()
    envmapIndDl_cpu = dataBatch['envmapDlInd']
    envmapIndDlBatch = (envmapIndDl_cpu ).cuda()
    envmapIndDm_cpu = dataBatch['envmapDmInd']
    envmapIndDmBatch = (envmapIndDm_cpu ).cuda()
    envmapIndBatch = torch.cat([envmapIndBatch, \
            envmapIndDlBatch, envmapIndDmBatch], dim=0 )

    return envmapBatch, envmapIndBatch

class BatchLoader(Dataset):
    def __init__(self, dataRoot,
            imHeight = 240, imWidth = 320,
            phase='TRAIN', rseed = None,
            isShading = False,
            isLight = False,
            isLightSrc = False,
            isDepthShadow = False,
            envHeight = 8, envWidth = 16,
            envRow = 120, envCol = 160,
            maxWinNum = 3, maxLampNum = 7,
            isPredDepth = False ):

        self.imHeight = imHeight
        self.imWidth = imWidth
        self.phase = phase.upper()

        self.isShading = isShading
        self.isLight = isLight
        self.isLightSrc = isLightSrc

        self.envWidth = envWidth
        self.envHeight = envHeight
        self.envRow = envRow
        self.envCol = envCol
        self.envWidth = envWidth
        self.envHeight = envHeight
        self.maxLampNum = maxLampNum
        self.maxWinNum = maxWinNum

        self.isDepthShadow = isDepthShadow

        # Load shapes
        shapeList = []
        dirs = ['main_xml', 'main_xml1']
        if phase.upper() == 'TRAIN':
            sceneFile = osp.join(dataRoot, 'train.txt')
        elif phase.upper() == 'TEST':
            sceneFile = osp.join(dataRoot, 'test.txt')
        else:
            print('Wrong: unrecognizable phase')
            assert(False )

        with open(sceneFile, 'r') as fIn:
            sceneList = fIn.readlines()
        sceneList = [x.strip() for x in sceneList ]

        for d in dirs:
            shapeList = shapeList + [osp.join(dataRoot, d, x) for x in sceneList ]
        shapeList = sorted(shapeList )
        print('Shape Num: %d' % len(shapeList ) )

        self.imList = []
        for shape in shapeList:
            imNames = sorted(glob.glob(osp.join(shape, 'im_*.hdr') ) )
            self.imList = self.imList + imNames

        self.imDlList = [x.replace('main_', 'mainDiffLight_') for x in self.imList ]
        self.imDmList = [x.replace('main_', 'mainDiffMat_') for x in self.imList ]

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

        if self.isShading:
            self.shadingDirectList = [x.replace('im_', 'imshadingDirect_').replace('.hdr', '.rgbe') for x in self.imList ]
            self.shadingDirectDlList = [x.replace('main_', 'mainDiffLight_') for x in self.shadingDirectList ]

            self.shadingList = [x.replace('im_', 'imshading_') for x in self.imList ]
            self.shadingDlList = [x.replace('main_', 'mainDiffLight_') for x in self.shadingList ]
            self.shadingDmList = [x.replace('main_', 'mainDiffMat_') for x in self.shadingList ]

        if self.isLight:
            self.envList = [x.replace('im_', 'imenv_') for x in self.imList ]
            self.envDlList = [x.replace('main_', 'mainDiffLight_') for x in self.envList ]
            self.envDmList = [x.replace('main_', 'mainDiffMat_') for x in self.envList ]

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

            if self.isDepthShadow:
                self.depthShadowList = []

            for x in self.imList:
                lightDir = x.replace('im_', 'light_').replace('.hdr', '')
                lightMaskNames = glob.glob(osp.join(lightDir, 'mask*.png') )
                lightBoxNames = [x.replace('mask', 'box').replace('.png', '.dat')
                        for x in lightMaskNames ]
                lightShadingNames = [x.replace('mask', 'imDS').replace('.png', '.rgbe')
                        for x in lightMaskNames ]
                lightShadingDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightShadingNames ]

                lightShadingNoNames = [x.replace('mask', 'imDSNoOcclu').replace('.png', '.rgbe')
                        for x in  lightMaskNames ]
                lightShadingNoDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightShadingNoNames ]

                lightShadingNoAppNames = [x.replace('mask', 'imDSNoOccluApprox').replace('.png', '.hdr')
                        for x in  lightMaskNames ]
                lightShadingNoAppDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightShadingNoAppNames ]

                lightShadowNames = [x.replace('mask', 'imShadow') for x in lightMaskNames ]

                lightSrcNames = [x.replace('mask', 'lightSrc').replace('.png', '.dat')
                        for x in lightMaskNames ]
                lightSrcDlNames = [x.replace('main_', 'mainDiffLight_') for x in lightSrcNames ]


                self.lightMaskList.append(lightMaskNames )
                self.lightBoxList.append(lightBoxNames )

                self.lightShadingList.append(lightShadingNames )
                self.lightShadingDlList.append(lightShadingDlNames )

                self.lightShadingNoList.append(lightShadingNoNames )
                self.lightShadingNoDlList.append(lightShadingNoDlNames )

                self.lightShadingNoAppList.append(lightShadingNoAppNames )
                self.lightShadingNoAppDlList.append(lightShadingNoAppDlNames )

                self.lightSrcList.append(lightSrcNames )
                self.lightSrcDlList.append(lightSrcDlNames )

                self.lightShadowList.append(lightShadowNames )

                if self.isDepthShadow:
                    depthShadowNames = [x.replace('mask', 'depthShadow').replace('.png', '.npy') for x in lightMaskNames ]
                    self.depthShadowList.append(depthShadowNames )


        # Permute the image list
        self.count = len(self.imList )
        self.perm = list(range(self.count ) )

        if rseed is not None:
            random.seed(rseed )

        if self.phase.upper() == 'TRAIN':
            random.shuffle(self.perm )

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

        if self.isShading == True:
            shadingDirect = self.loadHdr(self.shadingDirectList[self.perm[ind] ], self.envRow, self.envCol )
            shadingDirectDl = self.loadHdr(self.shadingDirectDlList[self.perm[ind] ], self.envRow, self.envCol )
            shadingDirectDm = shadingDirect
            shadingDirect = shadingDirect * scale
            shadingDirectDl = shadingDirectDl * scaleDl
            shadingDirectDm = shadingDirectDm * scaleDm

            shading = self.loadHdr( self.shadingList[self.perm[ind] ], self.envRow, self.envCol )
            shadingDl = self.loadHdr( self.shadingDlList[self.perm[ind] ], self.envRow, self.envCol )
            shadingDm = self.loadHdr( self.shadingDmList[self.perm[ind] ], self.envRow, self.envCol )
            shading = shading * scale
            shadingDl = shadingDl * scaleDl
            shadingDm = shadingDm * scaleDm

        if self.isLight == True:
            envmap, envmapInd = self.loadEnvmap(self.envList[self.perm[ind] ] )
            envmap = envmap * scale

            envmapDl, envmapDlInd = self.loadEnvmap(self.envDlList[self.perm[ind] ] )
            envmapDl = envmapDl * scaleDl

            envmapDm, envmapDmInd = self.loadEnvmap(self.envDmList[self.perm[ind] ] )
            envmapDm = envmapDm * scaleDm

        if self.isLightSrc == True:
            lightMaskNames = self.lightMaskList[self.perm[ind ] ]
            lightBoxNames = self.lightBoxList[self.perm[ind ] ]

            shadingNames = self.lightShadingList[self.perm[ind ] ]
            shadingDlNames = self.lightShadingDlList[self.perm[ind ] ]

            shadingNoNames = self.lightShadingNoList[self.perm[ind ] ]
            shadingNoDlNames = self.lightShadingNoDlList[self.perm[ind ] ]

            shadingNoAppNames = self.lightShadingNoAppList[self.perm[ind ] ]
            shadingNoAppDlNames = self.lightShadingNoAppDlList[self.perm[ind ] ]

            shadowNames = self.lightShadowList[self.perm[ind ] ]
            if self.isDepthShadow:
                depthShadowNames = self.depthShadowList[self.perm[ind ] ]

            lightSrcNames = self.lightSrcList[self.perm[ind ] ]
            lightSrcDlNames = self.lightSrcDlList[self.perm[ind ] ]

            # Visible lamp light
            visLampMask = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth ), dtype = np.float32 )
            onLampMask = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth), dtype = np.float32 )
            onLampMaskDl = np.zeros( (self.maxLampNum, self.imHeight, self.imWidth), dtype = np.float32 )

            visLampCenter = np.zeros( (self.maxLampNum, 3 ), dtype = np.float32 )
            visLampAxes = np.zeros( (self.maxLampNum, 3, 3 ), dtype = np.float32 )

            visLampShading = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visLampShadingDl = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visLampShadingDm = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visLampShadingNo = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visLampShadingNoDl = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visLampShadingNoDm = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
            if self.isDepthShadow:
                depthVisLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
                confVisLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )

            visLampNum = np.zeros( (1 ), dtype = np.int32 )

            # Visible window Light
            visWinMask = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth ), dtype = np.float32 )
            onWinMask = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth), dtype = np.float32 )
            onWinMaskDl = np.zeros( (self.maxWinNum, self.imHeight, self.imWidth), dtype = np.float32 )

            visWinPlane = np.zeros( (self.maxWinNum, 12 ), dtype = np.float32 )

            visWinSrc = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            visWinSrcDl = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            visWinSrcDm = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )

            visWinShading = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visWinShadingNo = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visWinShadingNoApp = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoAppDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            visWinShadingNoAppDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            visWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
            if self.isDepthShadow:
                depthVisWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
                confVisWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )

            visWinNum = np.zeros( (1 ), dtype = np.int32 )

            # Invisible lamp light
            invLampCenter = np.zeros( (self.maxLampNum, 3 ), dtype = np.float32 )
            invLampAxes = np.zeros( (self.maxLampNum, 3, 3), dtype = np.float32 )
            invLampOn = np.zeros(self.maxLampNum, dtype=np.float32 )
            invLampOnDl = np.zeros(self.maxLampNum, dtype=np.float32 )

            invLampShading = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )
            invLampShadingDl = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )
            invLampShadingDm = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )

            invLampShadingNo = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )
            invLampShadingNoDl = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )
            invLampShadingNoDm = np.zeros( (self.maxLampNum, 3, self.envRow, self.envCol ), dtype=np.float32 )

            invLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
            if self.isDepthShadow:
                depthInvLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
                confInvLampShadow = np.zeros( (self.maxLampNum, 1, self.envRow, self.envCol ), dtype = np.float32 )

            invLampNum = np.zeros( (1 ), dtype = np.int32 )

            # Invisible window light
            invWinPlane = np.zeros( (self.maxWinNum, 12 ), dtype = np.float32 )

            invWinSrc = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            invWinSrcDl = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )
            invWinSrcDm = np.zeros( (self.maxWinNum, 22 ), dtype = np.float32 )

            invWinOn = np.zeros(self.maxWinNum, dtype=np.float32 )
            invWinOnDl = np.zeros(self.maxWinNum, dtype=np.float32 )

            invWinShading = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            invWinShadingNo = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingNoDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingNoDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            invWinShadingNoApp = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingNoAppDl = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )
            invWinShadingNoAppDm = np.zeros( (self.maxWinNum, 3, self.envRow, self.envCol), dtype=np.float32 )

            invWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
            if self.isDepthShadow:
                depthInvWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )
                confInvWinShadow = np.zeros( (self.maxWinNum, 1, self.envRow, self.envCol ), dtype = np.float32 )

            invWinNum = np.zeros( (1 ), dtype = np.int32 )

            # raw parameters for light source
            for n in range(0, len(lightMaskNames ) ):
                maskName = lightMaskNames[n ]
                boxName = lightBoxNames[n ]

                shName = shadingNames[n ]
                shDlName = shadingDlNames[n ]

                shNoName = shadingNoNames[n ]
                shNoDlName = shadingNoDlNames[n ]

                shNoAppName = shadingNoAppNames[n ]
                shNoAppDlName = shadingNoAppDlNames[n ]

                shdName = shadowNames[n ]
                if self.isDepthShadow:
                    depthShdName = depthShadowNames[n ]

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

                lightShading = self.loadHdr(shName, self.envRow, self.envCol )
                lightShadingDl = self.loadHdr(shDlName, self.envRow, self.envCol )

                lightShadingNo = self.loadHdr(shNoName, self.envRow, self.envCol )
                lightShadingNoDl = self.loadHdr(shNoDlName, self.envRow, self.envCol )

                lightShadow = cv2.imread(shdName )[:, :, 0]
                if lightShadow.shape[0] != self.envRow:
                    lightShadow = cv2.resize(lightShadow, (self.envRow, self.envCol ),
                                             interpolation = cv2.INTER_AREA )
                lightShadow = (lightShadow.astype(np.float32 ) / 255.0)[np.newaxis, :]
                if self.isDepthShadow:
                    depthShadow = np.load(depthShdName ).squeeze()
                    confShadow = (depthShadow < 0)
                    confShadow = ndimage.binary_dilation(confShadow, structure = np.ones( (5, 5) ) )
                    confShadow = confShadow.astype(np.float32 )
                    confShadow[0:3, :] = 1.0
                    confShadow[:, 0:3] = 1.0
                    confShadow[-3:, :] = 1.0
                    confShadow[:, -3:] = 1.0
                    if depthShadow.shape[0] != self.envRow:
                        depthShadow = cv2.resize(depthShadow, (self.envRow, self.envCol ),
                                                 interpolation = cv2.INTER_AREA )
                        confShadow = cv2.resize(confShadow, (self.envRow, self.envCol ),
                                                interpolation = cv2.INTER_AREA )
                    depthShadow = (depthShadow.astype(np.float32 ) )[np.newaxis, :]
                    confShadow = (confShadow.astype(np.float32) )[np.newaxis, :]

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

                        visWinShading[visWinNum[0], ] = lightShading * scale
                        visWinShadingDl[visWinNum[0], ] = lightShadingDl * scaleDl
                        visWinShadingDm[visWinNum[0], ] = lightShading * scaleDm

                        visWinShadingNo[visWinNum[0], ] = lightShadingNo * scale
                        visWinShadingNoDl[visWinNum[0], ] = lightShadingNoDl * scaleDl
                        visWinShadingNoDm[visWinNum[0], ] = lightShadingNo * scaleDm

                        visWinShadow[visWinNum[0], ] = lightShadow
                        if self.isDepthShadow:
                            depthVisWinShadow[visWinNum[0], ] = depthShadow
                            confVisWinShadow[visWinNum[0], ] = confShadow

                        visWinNum[0] = visWinNum[0] + 1
                    else:
                        visLampMask[visLampNum[0], :] = mask
                        if np.sum(intensity ) > 0:
                            onLampMask[visLampNum[0], :] = mask
                        if np.sum(intensityDl ) > 0:
                            onLampMaskDl[visLampNum[0], :] = mask

                        axes = []
                        axes.append(xAxis.reshape(1, 3) )
                        axes.append(yAxis.reshape(1, 3) )
                        axes.append(zAxis.reshape(1, 3) )
                        axes = np.concatenate(axes, axis=0 )

                        visLampCenter[visLampNum[0], :]  = center
                        visLampAxes[visLampNum[0], :] = axes

                        visLampShading[visLampNum[0], ] = lightShading * scale
                        visLampShadingDl[visLampNum[0], ] = lightShadingDl * scaleDl
                        visLampShadingDm[visLampNum[0], ] = lightShading * scaleDm

                        visLampShadingNo[visLampNum[0], ] = lightShadingNo * scale
                        visLampShadingNoDl[visLampNum[0], ] = lightShadingNoDl * scaleDl
                        visLampShadingNoDm[visLampNum[0], ] = lightShadingNo * scaleDm

                        visLampShadow[visLampNum[0], ] = lightShadow
                        if self.isDepthShadow:
                            depthVisLampShadow[visLampNum[0], ] = depthShadow
                            confVisLampShadow[visLampNum[0], ] = confShadow

                        visLampNum[0] = visLampNum[0] + 1
                else:
                    if boxPara['isWindow']:
                        if np.sum(intensity ) > 0:
                            invWinOn[invWinNum[0] ] = 1

                            lightShadingNoApp = self.loadHdr(shNoAppName, self.envRow, self.envCol ) * scale
                            lightShadingNoAppDm = lightShadingNoApp / scale * scaleDm

                            invWinShadingNoApp[invWinNum[0], ] = lightShadingNoApp
                            invWinShadingNoAppDm[invWinNum[0], ] = lightShadingNoAppDm

                            winSrc[-1]  = srcPara['loss'] / srcPara['scale'] * scale
                            winSrcDm[-1] = srcPara['loss'] / srcPara['scale'] * scaleDm

                        if np.sum(intensityDl ) > 0:
                            invWinOnDl[invWinNum[0] ] = 1

                            lightShadingNoAppDl = self.loadHdr(shNoAppDlName, self.envRow, self.envCol ) * scaleDl
                            invWinShadingNoAppDl[invWinNum[0], :] = lightShadingNoAppDl

                            winSrcDl[-1] = srcDlPara['loss'] / srcDlPara['scale'] * scaleDl

                        invWinSrc[invWinNum[0], :] = winSrc
                        invWinSrcDl[invWinNum[0], :] = winSrcDl
                        invWinSrcDm[invWinNum[0], :] = winSrcDm


                        # Get the center and the points
                        winCenter = center
                        invWinPlane[invWinNum[0] ] = np.concatenate(
                                [winCenter, zAxisNormalized, yAxis, xAxis], axis=0 )

                        invWinShading[invWinNum[0], ] = lightShading * scale
                        invWinShadingDl[invWinNum[0], ] = lightShadingDl * scaleDl
                        invWinShadingDm[invWinNum[0], ] = lightShading * scaleDm

                        invWinShadingNo[invWinNum[0], ] = lightShadingNo * scale
                        invWinShadingNoDl[invWinNum[0], ] = lightShadingNoDl * scaleDl
                        invWinShadingNoDm[invWinNum[0], ] = lightShadingNo * scaleDm

                        invWinShadow[invWinNum[0], ] = lightShadow
                        if self.isDepthShadow:
                            depthInvWinShadow[invWinNum[0], ] = depthShadow
                            confInvWinShadow[invWinNum[0], ] = confShadow

                        invWinNum[0] = invWinNum[0] + 1
                    else:
                        if np.sum(intensity ) > 0:
                            invLampOn[invLampNum[0] ] = 1
                        if np.sum(intensityDl ) > 0:
                            invLampOnDl[invLampNum[0] ] = 1

                        axes = []
                        axes.append(xAxis.reshape(1, 3) )
                        axes.append(yAxis.reshape(1, 3) )
                        axes.append(zAxis.reshape(1, 3) )
                        axes = np.concatenate(axes, axis=0 )
                        invLampCenter[invLampNum[0], :]  = center
                        invLampAxes[invLampNum[0], :]  = axes

                        invLampShading[invLampNum[0], ] = lightShading * scale
                        invLampShadingDl[invLampNum[0], ] = lightShadingDl * scaleDl
                        invLampShadingDm[invLampNum[0], ] = lightShading * scaleDm

                        invLampShadingNo[invLampNum[0], ] = lightShadingNo * scale
                        invLampShadingNoDl[invLampNum[0], ] = lightShadingNoDl * scaleDl
                        invLampShadingNoDm[invLampNum[0], ] = lightShadingNo * scaleDm

                        invLampShadow[invLampNum[0], ] = lightShadow
                        if self.isDepthShadow:
                            depthInvLampShadow[invLampNum[0], ] = depthShadow
                            confInvLampShadow[invLampNum[0], ] = confShadow

                        invLampNum[0] = invLampNum[0] + 1

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

        if self.isShading:
            batchDict['shading'] = shading
            batchDict['shadingDl'] = shadingDl
            batchDict['shadingDm'] = shadingDm

            batchDict['shadingDirect'] = shadingDirect
            batchDict['shadingDirectDl'] = shadingDirectDl
            batchDict['shadingDirectDm'] = shadingDirectDm

        if self.isLight:
            batchDict['envmap'] = envmap
            batchDict['envmapInd'] = envmapInd

            batchDict['envmapDl'] = envmapDl
            batchDict['envmapDlInd'] = envmapDlInd

            batchDict['envmapDm'] = envmapDm
            batchDict['envmapDmInd'] = envmapDmInd

        if self.isLightSrc:
            lightMask = np.sum(visLampMask, axis=0, keepdims=True ) + np.sum(visWinMask, axis=0, keepdims=True )
            lightMask = np.clip(lightMask, 0, 1 )
            lightOnMask = np.sum(onLampMask, axis=0, keepdims=True) + np.sum(onWinMask, axis=0, keepdims=True )
            lightOnMask = np.clip(lightOnMask, 0, 1 )
            lightOnMaskDl = np.sum(onLampMaskDl, axis=0, keepdims=True) + np.sum(onWinMaskDl, axis=0, keepdims=True )
            lightOnMaskDl = np.clip(lightOnMaskDl, 0, 1 )
            batchDict['lightMasks'] = lightMask
            batchDict['lightOnMasks'] = lightOnMask
            batchDict['lightOnMasksDl'] = lightOnMaskDl

            # Visible light lamp
            batchDict['lampMasks'] = visLampMask
            batchDict['onLampMasks'] = onLampMask
            batchDict['onLampMasksDl'] = onLampMaskDl

            batchDict['visLampCenters'] = visLampCenter
            batchDict['visLampAxes'] = visLampAxes

            batchDict['visLampShadings'] = visLampShading
            batchDict['visLampShadingsDl'] = visLampShadingDl
            batchDict['visLampShadingsDm'] = visLampShadingDm
            batchDict['visLampShadingsNo'] = visLampShadingNo
            batchDict['visLampShadingsNoDl'] = visLampShadingNoDl
            batchDict['visLampShadingsNoDm'] = visLampShadingNoDm

            batchDict['visLampShadows'] = visLampShadow
            if self.isDepthShadow:
                batchDict['depthVisLampShadows'] = depthVisLampShadow
                batchDict['confVisLampShadows'] = confVisLampShadow

            batchDict['visLampNum'] = visLampNum

            # Visible light window
            batchDict['winMasks'] = visWinMask
            batchDict['onWinMasks'] = onWinMask
            batchDict['onWinMasksDl'] = onWinMaskDl

            batchDict['visWinPlanes'] = visWinPlane

            batchDict['visWinSrcs'] = visWinSrc
            batchDict['visWinSrcsDl'] = visWinSrcDl
            batchDict['visWinSrcsDm'] = visWinSrcDm

            batchDict['visWinShadings'] = visWinShading
            batchDict['visWinShadingsDl'] = visWinShadingDl
            batchDict['visWinShadingsDm'] = visWinShadingDm

            batchDict['visWinShadingsNo'] = visWinShadingNo
            batchDict['visWinShadingsNoDl'] = visWinShadingNoDl
            batchDict['visWinShadingsNoDm'] = visWinShadingNoDm

            batchDict['visWinShadingsNoApp'] = visWinShadingNoApp
            batchDict['visWinShadingsNoAppDl'] = visWinShadingNoAppDl
            batchDict['visWinShadingsNoAppDm'] = visWinShadingNoAppDm

            batchDict['visWinShadows'] = visWinShadow
            if self.isDepthShadow:
                batchDict['depthVisWinShadows'] = depthVisWinShadow
                batchDict['confVisWinShadows'] = confVisWinShadow

            batchDict['visWinNum'] = visWinNum

            # Invisible light lamp
            batchDict['invLampOn'] = invLampOn
            batchDict['invLampOnDl'] = invLampOnDl

            batchDict['invLampShadings'] = invLampShading
            batchDict['invLampShadingsDl'] = invLampShadingDl
            batchDict['invLampShadingsDm'] = invLampShadingDm
            batchDict['invLampShadingsNo'] = invLampShadingNo
            batchDict['invLampShadingsNoDl'] = invLampShadingNoDl
            batchDict['invLampShadingsNoDm'] = invLampShadingNoDm

            batchDict['invLampShadows'] = invLampShadow
            if self.isDepthShadow:
                batchDict['depthInvLampShadows'] = depthInvLampShadow
                batchDict['confInvLampShadows'] = confInvLampShadow

            batchDict['invLampCenters'] = invLampCenter
            batchDict['invLampAxes'] = invLampAxes
            batchDict['invLampNum'] = invLampNum

            # Invisible light window
            batchDict['invWinOn'] = invWinOn
            batchDict['invWinOnDl'] = invWinOnDl

            batchDict['invWinPlanes'] = invWinPlane

            batchDict['invWinSrcs'] = invWinSrc
            batchDict['invWinSrcsDl'] = invWinSrcDl
            batchDict['invWinSrcsDm'] = invWinSrcDm

            batchDict['invWinShadings'] = invWinShading
            batchDict['invWinShadingsDl'] = invWinShadingDl
            batchDict['invWinShadingsDm'] = invWinShadingDm

            batchDict['invWinShadingsNo'] = invWinShadingNo
            batchDict['invWinShadingsNoDl'] = invWinShadingNoDl
            batchDict['invWinShadingsNoDm'] = invWinShadingNoDm

            batchDict['invWinShadingsNoApp'] = invWinShadingNoApp
            batchDict['invWinShadingsNoAppDl'] = invWinShadingNoAppDl
            batchDict['invWinShadingsNoAppDm'] = invWinShadingNoAppDm

            batchDict['invWinShadows'] = invWinShadow
            if self.isDepthShadow:
                batchDict['depthInvWinShadows'] = depthInvWinShadow
                batchDict['confInvWinShadows'] = confInvWinShadow

            batchDict['invWinNum'] = invWinNum

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

    def loadEnvmap(self, envName ):
        if not osp.isfile(envName ):
            print(envName )
            env = np.zeros((3, self.envRow, self.envCol, self.envHeight,
                self.envWidth ), dtype=np.float32 )
            envInd = np.zeros((1, 1, 1), dtype=np.float32 )
            return env, envInd
        else:
            envHeightOrig, envWidthOrig = 16, 32
            assert( (envHeightOrig / self.envHeight) == (envWidthOrig / self.envWidth) )
            assert( envHeightOrig % self.envHeight == 0)

            env = cv2.imread(envName, -1 )
            if not env is None:
                env = env.reshape(self.envRow, envHeightOrig, self.envCol,
                    envWidthOrig, 3)
                env = np.ascontiguousarray(env.transpose([4, 0, 2, 1, 3] ) )

                scale = int(envHeightOrig / self.envHeight )
                if scale > 1:
                    env = block_reduce(env, block_size = (1, 1, 1, scale, scale), func = np.mean )

                envInd = np.ones((1, 1, 1), dtype=np.float32 )
                return env, envInd
            else:
                print(envName )
                env = np.zeros((3, self.envRow, self.envCol, self.envHeight,
                    self.envWidth ), dtype=np.float32 )
                envInd = np.zeros((1, 1, 1), dtype=np.float32 )
                return env, envInd
