"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

import numpy as np

#from util.misc import visualize_attention

offset = 0.189
scale = 0.515

import os.path as osp
curDir = '/'.join(osp.abspath(__file__).split('/')[0:-1] )



def run(input_names, output_names, mask_names,
        model_type="dpt_hybrid", optimize=True ):
    print("initialize")

    if model_type == 'dpt_hybrid':
        model_path = osp.join(curDir, 'weights', 'dpt_hybrid.pt' )
    elif model_type == 'dpt_large':
        model_path = osp.join(curDir, 'weights', 'dpt_large.pt' )

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large":  # DPT-Large
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid":  # DPT-Hybrid
        net_w = net_h = 384
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        assert (
            False
        ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()

    if optimize == True and device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device )

    print("start processing")

    for n in range(0, len(input_names ) ):
        input_name = input_names[n]
        mask_name = mask_names[n]
        output_name = output_names[n]

        img = util.io.read_image(input_name)
        img_input = transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

            if optimize == True and device == torch.device("cuda"):
                sample = sample.to(memory_format=torch.channels_last)
                sample = sample.half()

            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        mask = cv2.imread(mask_name )
        if len(mask.shape ) == 3:
            mask = mask[:, :, 0]
        mask = (1 - mask.astype(np.float32 ) / 255.0 )

        predictionMin = (prediction + mask * 1e6).min()
        predictionMax = (prediction - mask * 1e6).max()


        prediction = (prediction - predictionMin ) \
            / (predictionMax - predictionMin )
        prediction = np.clip(prediction, 0, 1)

        prediction = 1 / (scale * prediction + offset)
        prediction = prediction * (1 - mask ) + mask * (1.420 + 3.869 )

        np.save(output_name, prediction )

        print("finished")
