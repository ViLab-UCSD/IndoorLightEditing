import torch
import utils
import models
import numpy as np
import pytorch3d.loss as torch3dLoss


def gradLoss(pred, gt, mask, gaps = [1, 2, 4, 8, 16],
             isOverlap = True, isScaleInvariant = False ):
    errs = 0
    for n in range(0, len(gaps ) ):
        gap = gaps[n]

        if isScaleInvariant:
            pred_x = (pred[:, :, :, gap:] - pred[:, :, :, :-gap ] ) \
                / torch.clamp(pred[:, :, :, gap:] + pred[:, :, :, :-gap ], min=1e-7 )
            gt_x = (gt[:, :, :, gap:] - gt[:, :, :, :-gap ]  ) \
                / torch.clamp(gt[:, :, :, gap:] + gt[:, :, :, :-gap ], min=1e-7 )
        else:
            pred_x = pred[:, :, :, gap:] - pred[:, :, :, :-gap ]
            gt_x = gt[:, :, :, gap:] - gt[:, :, :, :-gap ]

        if isOverlap:
            mask_x = mask[:, :, :, gap:] * mask[:, :, :, :-gap]
        else:
            mask_x = torch.clamp(mask[:, :, :, gap:] + mask[:, :, :, :-gap], 0, 1 )

        err_x = torch.mean(torch.pow(pred_x - gt_x, 2 ) * mask_x )

        if isScaleInvariant:
            pred_y = (pred[:, :, gap:, :] - pred[:, :, :-gap, :] ) \
                / torch.clamp(pred[:, :, gap:, :] + pred[:, :, :-gap, :], min=1e-7 )
            gt_y = (gt[:, :, gap:, :] - gt[:, :, :-gap, :] ) \
                / torch.clamp(gt[:, :, gap:, :] + gt[:, :, :-gap, :], min=1e-7 )
        else:
            pred_y = pred[:, :, gap:, :] - pred[:, :, :-gap, :]
            gt_y = gt[:, :, gap:, :] - gt[:, :, :-gap, :]

        if isOverlap:
            mask_y = mask[:, :, gap:, :] * mask[:, :, :-gap, :]
        else:
            mask_y = torch.clamp(mask[:, :, gap:, :] + mask[:, :, :-gap, :], 0, 1 )

        err_y = torch.mean(torch.pow(pred_y - gt_y, 2 ) * mask_y )

        errs += (err_x + err_y )

    errs /= len(gaps )

    return errs

def samplePointsFromBoxSurf(center, axes, sampleNum ):
    z1 = np.linspace(-0.5, 0.5, sampleNum ).astype(np.float32 )
    z2 = np.linspace(-0.5, 0.5, sampleNum ).astype(np.float32 )
    xP, yP = np.meshgrid(z1, z2 )
    xP = xP.reshape(1, sampleNum * sampleNum, 1)
    yP = yP.reshape(1, sampleNum * sampleNum, 1)
    xP = torch.from_numpy(xP ).cuda()
    yP = torch.from_numpy(yP ).cuda()

    points_1 = center + xP * axes[:, 0:1, :] \
            + yP * axes[:, 1:2, :] \
            + 0.5 * axes[:, 2:3, :]
    points_2 = center + xP * axes[:, 0:1, :] \
            + yP * axes[:, 1:2, :] \
            - 0.5 * axes[:, 2:3, :]
    points_3 = center + xP * axes[:, 0:1, :] \
            + 0.5 * axes[:, 1:2, :] \
            + yP * axes[:, 2:3, :]
    points_4 = center + xP * axes[:, 0:1, :] \
            - 0.5 * axes[:, 1:2, :] \
            + yP * axes[:, 2:3, :]
    points_5 = center + 0.5 * axes[:, 0:1, :] \
            + xP * axes[:, 1:2, :] \
            + yP * axes[:, 2:3, :]
    points_6 = center - 0.5 * axes[:, 0:1, :] \
            + xP * axes[:, 1:2, :] \
            + yP * axes[:, 2:3, :]
    points = torch.cat([
        points_1, points_2, points_3,
        points_4, points_5, points_6 ], dim=1 )

    return points


def samplePointsFromPlane(center, xAxis, yAxis, sampleNum ):
    z1 = np.linspace(-0.5, 0.5, sampleNum ).astype(np.float32 )
    z2 = np.linspace(-0.5, 0.5, sampleNum ).astype(np.float32 )
    xP, yP = np.meshgrid(z1, z2 )
    xP = xP.reshape(1, sampleNum * sampleNum, 1 )
    yP = yP.reshape(1, sampleNum * sampleNum, 1 )
    xP = torch.from_numpy(xP ).cuda()
    yP = torch.from_numpy(yP ).cuda()

    points = center + xAxis * xP + yAxis * yP

    return points

def computeWinSrcErr(srcGt, srcPred):
    intGt, axisGt, lambGt = torch.split(srcGt, [3, 3, 1], dim=0 )
    intPred, axisPred, lambPred = torch.split(srcPred, [3, 3, 1], dim=0 )

    intErr = torch.mean(torch.pow(torch.log(intGt + 1) - torch.log(intPred + 1), 2) )
    axisErr = 1 - torch.sum(axisPred * axisGt )
    lambErr = torch.mean(torch.pow(torch.log(lambGt + 1) - torch.log(lambPred + 1), 2) )

    return intErr, axisErr, lambErr


def visLampLoss(
        visLampNum,
        visLampPointsPreds,
        visLampOns,
        visLampAxesBatch, visLampCentersBatch,
        depthScaleBatch = None,
        sampleNum = 4, isTest = False ):

    visLampPointsErr = 0
    if visLampNum > 0:
        batchSize = visLampCentersBatch.size(0 )
        for m in range(0, batchSize ):
            for n in range(0, visLampNum ):
                if visLampOns[m, n ] == 1:
                    visLampPointsPred = visLampPointsPreds[m * visLampNum + n ]

                    visLampCenterGt = visLampCentersBatch[m:m+1, n:n+1, :]
                    visLampAxesGt = visLampAxesBatch[m:m+1, n, :]
                    visLampPointsGt = samplePointsFromBoxSurf(
                            visLampCenterGt, visLampAxesGt, sampleNum )

                    dist, _ = torch3dLoss.chamfer_distance(visLampPointsPred, visLampPointsGt, isRMSE=True )

                    if isTest:
                        visLampPointsErr += (dist ) * depthScaleBatch[m]
                    else:
                        visLampPointsErr += dist

    return visLampPointsErr



def visWindowLoss(
        visWinNum,
        visWinCenterPreds, visWinNormalPreds,
        visWinXAxisPreds, visWinYAxisPreds,
        visWinSrcPreds, visWinSrcSkyPreds, visWinSrcGrdPreds,
        visWinOns,
        visWinPlanesBatch, visWinSrcsBatch,
        depthScaleBatch = None,
        sampleNum = 10, isTest = False ):


    visWinPointsErr = 0
    visWinNormalErr = 0
    visWinSizeErr = 0
    visWinSrcErr = [0, 0, 0]
    visWinSrcSkyErr =[0, 0, 0]
    visWinSrcGrdErr = [0, 0, 0]

    if visWinNum > 0:
        batchSize = visWinPlanesBatch.size(0 )
        for m in range(0, batchSize ):
            for n in range(0, visWinNum ):
                if visWinOns[m, n].item() == 1:
                    # Sample predicted points
                    visWinCenterPred = visWinCenterPreds[m * visWinNum + n].unsqueeze(1)
                    visWinYAxisPred = visWinYAxisPreds[m * visWinNum + n].unsqueeze(1)
                    visWinXAxisPred = visWinXAxisPreds[m * visWinNum + n].unsqueeze(1)
                    winPointsPred = samplePointsFromPlane(
                            visWinCenterPred, visWinYAxisPred, visWinXAxisPred, sampleNum)

                    # Sample groundtruth points
                    visWinCenterGt = visWinPlanesBatch[m:m+1, n:n+1, 0:3]
                    visWinYAxisGt = visWinPlanesBatch[m:m+1, n:n+1, 6:9]
                    visWinXAxisGt = visWinPlanesBatch[m:m+1, n:n+1, 9:12]
                    winPointsGt = samplePointsFromPlane(
                            visWinCenterGt, visWinYAxisGt, visWinXAxisGt, sampleNum )

                    dist, _ = torch3dLoss.chamfer_distance(winPointsPred, winPointsGt, isRMSE=True )

                    if isTest:
                        visWinPointsErr += (dist * depthScaleBatch[m] )
                    else:
                        visWinPointsErr += dist

                    visWinSizePred = torch.sqrt(torch.clamp(torch.sum(torch.pow(visWinXAxisPred, 2), dim=-1 ), min=1e-6 ) )  \
                        * torch.sqrt(torch.clamp(torch.sum(torch.pow(visWinYAxisPred, 2), dim=-1 ), min=1e-6 ) )
                    visWinSizeGt = torch.sqrt(torch.clamp(torch.sum(torch.pow(visWinXAxisGt, 2 ), dim=-1 ), min=1e-6 ) )\
                        * torch.sqrt(torch.clamp(torch.sum(torch.pow(visWinYAxisGt, 2 ), dim=-1 ), min=1e-6 ) )

                    if isTest:
                        visWinSizeErr += (torch.sum(torch.abs(visWinSizeGt - visWinSizePred ) ) * depthScaleBatch[m] )
                    else:
                        visWinSizeErr += torch.sum(torch.abs(visWinSizeGt - visWinSizePred ) )


                    # Normal
                    visWinNormalPred = visWinNormalPreds[m * visWinNum + n]
                    visWinNormalGt = visWinPlanesBatch[m:m+1, n, 3:6]
                    visWinNormalErr += 1 - torch.abs(torch.sum(
                        visWinNormalPred * visWinNormalGt ) )

                    visWinSrcGt = visWinSrcsBatch[m, n, 0:7 ]
                    visWinSrcSkyGt = visWinSrcsBatch[m, n, 7:14 ]
                    visWinSrcGrdGt = visWinSrcsBatch[m, n, 14:21 ]

                    visWinSrcPred = visWinSrcPreds[m * visWinNum + n].squeeze()
                    visWinSrcSkyPred = visWinSrcSkyPreds[m * visWinNum + n].squeeze()
                    visWinSrcGrdPred = visWinSrcGrdPreds[m * visWinNum + n].squeeze()

                    intErr, axisErr, lambErr = computeWinSrcErr(visWinSrcGt, visWinSrcPred )
                    visWinSrcErr[0] += intErr
                    visWinSrcErr[1] += axisErr
                    visWinSrcErr[2] += lambErr

                    intSkyErr, axisSkyErr, lambSkyErr = computeWinSrcErr(visWinSrcSkyGt, visWinSrcSkyPred )
                    visWinSrcSkyErr[0] += intSkyErr
                    visWinSrcSkyErr[1] += axisSkyErr
                    visWinSrcSkyErr[2] += lambSkyErr

                    intGrdErr, axisGrdErr, lambGrdErr = computeWinSrcErr(visWinSrcGrdGt, visWinSrcGrdPred )
                    visWinSrcGrdErr[0] += intGrdErr
                    visWinSrcGrdErr[1] += axisGrdErr
                    visWinSrcGrdErr[2] += lambGrdErr

    return visWinPointsErr, visWinNormalErr, visWinSizeErr, visWinSrcErr, visWinSrcSkyErr, visWinSrcGrdErr


def invLampLoss(
        invLampAxesPred, invLampCenterPred,
        invLampOns,
        invLampAxesGt, invLampCentersGt,
        depthScaleBatch = None,
        sampleNum = 4, isTest = False ):

    invLampPointsErr = 0
    invLampSizeErr = 0

    invLampPointsPred = samplePointsFromBoxSurf(
            invLampCenterPred.unsqueeze(1),
            invLampAxesPred, sampleNum )

    batchSize = invLampAxesPred.size(0 )
    for m in range(0, batchSize ):
        invLampOn = invLampOns[m]
        if invLampOn != 0:
            invLampCenterGt = invLampCentersGt[m].reshape(1, 1, 3)
            invLampAxisGt = invLampAxesGt[m].reshape(1, 3, 3 )

            invLampPointsGt = samplePointsFromBoxSurf(
                    invLampCenterGt, invLampAxisGt, sampleNum )

            dist, _  = torch3dLoss.chamfer_distance(invLampPointsPred[m:m+1, :], invLampPointsGt, isRMSE=True )
            if isTest:
                invLampPointsErr += (dist * depthScaleBatch[m] )
            else:
                invLampPointsErr += dist

            invLampSizePred = torch.prod(torch.sqrt(torch.clamp(torch.sum(
                invLampAxesPred[m, :, :] * invLampAxesPred[m, :, :], dim=-1), min=1e-6) ), dim=-1 )
            invLampSizeGt = torch.prod(torch.sqrt(torch.clamp(torch.sum(
                invLampAxisGt * invLampAxisGt, dim=-1 ), min=1e-6) ), dim=-1 )

            if isTest:
                invLampSizeErr += (torch.sum(torch.abs(invLampSizePred - invLampSizeGt ) ) * depthScaleBatch[m] )
            else:
                invLampSizeErr += torch.sum(torch.abs(invLampSizePred - invLampSizeGt ) )

    return  invLampPointsErr, invLampSizeErr


def invWindowLoss(
        invWinCenterPred, invWinNormalPred,
        invWinXAxisPred, invWinYAxisPred,
        invWinSrcPred, invWinSrcSkyPred, invWinSrcGrdPred,
        invWinOns,
        invWinPlanesGt, invWinSrcsGt,
        depthScaleBatch = None,
        sampleNum = 10, isTest = False ):

    invWinPointsErr = 0
    invWinNormalErr = 0
    invWinSizeErr = 0
    invWinSrcErr = [0, 0, 0]
    invWinSrcSkyErr = [0, 0, 0]
    invWinSrcGrdErr = [0, 0, 0]

    winPointsPred = samplePointsFromPlane(
            invWinCenterPred.unsqueeze(1),
            invWinYAxisPred.unsqueeze(1),
            invWinXAxisPred.unsqueeze(1),
            sampleNum  )

    batchSize = invWinCenterPred.size(0)
    for m in range(0, batchSize ):
        invWinOn = invWinOns[m]
        if invWinOn != 0:
            # Find out the best light sources
            invWinCenterGt = invWinPlanesGt[m, 0:3].reshape(1, 1, 3)
            invWinYAxisGt = invWinPlanesGt[m, 6:9].reshape(1, 1, 3)
            invWinXAxisGt = invWinPlanesGt[m, 9:12].reshape(1, 1, 3)

            winPointsGt = samplePointsFromPlane(
                    invWinCenterGt, invWinYAxisGt, invWinXAxisGt, sampleNum )

            dist, _ = torch3dLoss.chamfer_distance(winPointsPred[m:m+1, :], winPointsGt, isRMSE=True )

            if isTest:
                invWinPointsErr += (dist * depthScaleBatch[m ] )
            else:
                invWinPointsErr += dist


            invWinSizePred = torch.sqrt(torch.clamp(torch.sum(torch.pow(invWinXAxisPred[m, :], 2), dim=-1 ), min=1e-6 ) )\
                * torch.sqrt(torch.clamp(torch.sum(torch.pow(invWinYAxisPred[m, :], 2), dim=-1 ), min=1e-6 ) )
            invWinSizeGt = torch.sqrt(torch.clamp(torch.sum(torch.pow(invWinXAxisGt, 2 ), dim=-1 ), min=1e-6 ) ) \
                * torch.sqrt(torch.clamp(torch.sum(torch.pow(invWinYAxisGt, 2 ), dim=-1 ), min=1e-6 ) )

            if isTest:
                invWinSizeErr += (torch.sum(torch.abs(invWinSizeGt - invWinSizePred ) ) * depthScaleBatch[m ] )
            else:
                invWinSizeErr += torch.sum(torch.abs(invWinSizeGt - invWinSizePred ) )

            invWinNormalGt = invWinPlanesGt[m, 3:6]
            invWinNormalErr += 1 - torch.abs(torch.sum(
                invWinNormalPred[m, :] * invWinNormalGt ) )

            invWinSrcGt = invWinSrcsGt[m][0:7]
            invWinSrcSkyGt = invWinSrcsGt[m][7:14]
            invWinSrcGrdGt = invWinSrcsGt[m][14:21]

            intErr, axisErr, lambErr = computeWinSrcErr(invWinSrcGt, invWinSrcPred[m, :] )
            invWinSrcErr[0] += intErr
            invWinSrcErr[1] += axisErr
            invWinSrcErr[2] += lambErr

            intSkyErr, axisSkyErr, lambSkyErr = computeWinSrcErr(invWinSrcSkyGt, invWinSrcSkyPred[m, :] )
            invWinSrcSkyErr[0] += intSkyErr
            invWinSrcSkyErr[1] += axisSkyErr
            invWinSrcSkyErr[2] += lambSkyErr

            intGrdErr, axisGrdErr, lambGrdErr = computeWinSrcErr(invWinSrcGrdGt, invWinSrcGrdPred[m, :] )
            invWinSrcGrdErr[0] += intGrdErr
            invWinSrcGrdErr[1] += axisGrdErr
            invWinSrcGrdErr[2] += lambGrdErr

    return  invWinPointsErr, invWinNormalErr, invWinSizeErr, \
            invWinSrcErr, invWinSrcSkyErr, invWinSrcGrdErr


def lightLoss(envPred, shadingPred, envBatch, shadingBatch, segEnvBatch, sinWeight, lossType=0 ):
    pixelEnvNum = max( torch.sum(segEnvBatch ).cpu().data.item(), 1 )
    envHeight = envPred.size(-2)
    envWidth = envPred.size(-1)

    if lossType == 0:
        lightErr = torch.sum( ( torch.log(envPred+1) - torch.log(envBatch+1 ) )
            * ( torch.log(envPred+1)  - torch.log(envBatch+1 ) ) * segEnvBatch.unsqueeze(-1).unsqueeze(-1) * sinWeight ) \
                    / pixelEnvNum / 3.0 / envWidth / envHeight
    elif lossType == 1:
        lightErr = torch.sum( torch.abs(envPred - envBatch ) * segEnvBatch.unsqueeze(-1).unsqueeze(-1) * sinWeight ) \
                    / pixelEnvNum / 3.0 / envWidth / envHeight

    renderErr = torch.sum(
            (torch.log(shadingPred + 1) - torch.log(shadingBatch + 1) ) *
            (torch.log(shadingPred + 1) - torch.log(shadingBatch + 1) )
            * segEnvBatch ) / pixelEnvNum / 3.0

    return lightErr, renderErr



