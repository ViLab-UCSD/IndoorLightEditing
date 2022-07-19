import os
import os.path as osp
curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )
import sys
sys.path.append(osp.join(curDir, 'DPT' ) )
import argparse
import DPT.runRealMonoDepth as runDepth
import cv2
import numpy as np


parser = argparse.ArgumentParser()
# The directory of trained models
parser.add_argument('--experimentDepth', default='DPT', help='the path to store samples and models')
parser.add_argument('--mode', default='dpt_large', help='the mode for the DPT models' )
parser.add_argument('--testList', default=None, help='the path to store samples and models' )

# Starting and ending point
parser.add_argument('--rs', type=int, default=0, help='starting point' )
parser.add_argument('--re', type=int, default=100, help='ending point' )

opt = parser.parse_args()
print(opt )


with open(opt.testList, 'r') as fIn:
    dirList = fIn.readlines()
dirList = [x.strip() for x in dirList ]

input_names = []
mask_names = []
output_names = []

for dataId in range(max(opt.rs, 0), min(opt.re, len(dirList ) ) ):
    dataDir = dirList[dataId ]
    print(dataDir )
    inputDir = osp.join(dataDir, 'input')

    imName = osp.join(inputDir, 'im.png')
    im = cv2.imread(imName )
    height, width = im.shape[0:2]

    newHeight = int(np.round(float(height ) / 16  ) * 16 )
    newWidth = int(np.round(float(width ) / 16 ) * 16 )

    imNew = cv2.resize(im, (newWidth, newHeight ), interpolation = cv2.INTER_LINEAR )

    maskName = osp.join(inputDir, 'envMask.png' )
    envMask = cv2.imread(maskName )
    envMask = cv2.resize(envMask, (newWidth, newHeight ), interpolation = cv2.INTER_LINEAR )

    imNewName = imName.replace('im.png', 'imResized.png' )
    depthName = imName.replace('im.png', 'depthResized.npy' )
    maskNewName = imName.replace('im.png', 'envMaskResized.png' )

    cv2.imwrite(imNewName, imNew )
    cv2.imwrite(maskNewName, envMask )

    input_names.append(imNewName )
    mask_names.append(maskNewName )
    output_names.append(depthName )

runDepth.run(
        input_names,
        output_names,
        mask_names,
        model_type = opt.mode )

for depthName in output_names:
    depth = np.load(depthName )
    depth = depth.astype(np.float32 )

    depthNewName = depthName.replace('depthResized.npy', 'depth.npy' )
    imName = depthName.replace('depthResized.npy', 'im.png')

    im = cv2.imread(imName )
    height, width = im.shape[0:2]

    depth = cv2.resize(depth, (width, height), interpolation = cv2.INTER_LINEAR )
    np.save(depthNewName, depth )

    os.system('rm %s' % osp.join(depthName.replace('depth', '*').replace('.npy', '.*') ) )
