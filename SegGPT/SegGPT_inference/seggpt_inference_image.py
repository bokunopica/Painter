import os
import argparse

import torch
import numpy as np

from seggpt_engine import inference_image, inference_video
import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('SegGPT inference', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='seggpt_vit_large.pth')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='seggpt_vit_large_patch16_input896x448')
    parser.add_argument('--input_image', type=str, help='path to input image to be tested',
                        default=None)
    parser.add_argument('--input_video', type=str, help='path to input video to be tested',
                        default=None)
    parser.add_argument('--num_frames', type=int, help='number of prompt frames in video',
                        default=0)
    parser.add_argument('--prompt_image', type=str, nargs='+', help='path to prompt image',
                        default=None)
    parser.add_argument('--prompt_target', type=str, nargs='+', help='path to prompt target',
                        default=None)
    parser.add_argument('--seg_type', type=str, help='embedding for segmentation types', 
                        choices=['instance', 'semantic'], default='instance')
    parser.add_argument('--device', type=str, help='cuda or cpu',
                        default='cuda')
    parser.add_argument('--output_dir', type=str, help='path to output',
                        default='./')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='seggpt_vit_large_patch16_input896x448', seg_type='instance'):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    return model


if __name__ == '__main__':
    # args = get_args_parser()

    device = "cuda"
    ckpt_path = "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/pretrained_seggpt/seggpt_vit_large.pth"
    model = "seggpt_vit_large_patch16_input896x448"
    seg_type = "instance"
    # input_image = "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/examples/hmbb_3.jpg"
    input_image = "/mnt/c/data/breastCancer/Dataset_BUSI_with_GT/benign/benign (3).png"

    output_dir = "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/results"
    prompt_image = [
        # "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/examples/hmbb_1.jpg",
        # "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/examples/hmbb_2.jpg"
        "/mnt/c/data/breastCancer/Dataset_BUSI_with_GT/benign/benign (2).png"
    ]
    prompt_target = [
        # "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/examples/hmbb_1_target.png",
        # "/home/pico/myCodes/Painter/SegGPT/SegGPT_inference/examples/hmbb_2_target.png"
        "/mnt/c/data/breastCancer/Dataset_BUSI_with_GT/benign/benign (2)_mask.png"
    ]

    model = prepare_model(ckpt_path, model, seg_type).to(device)
    print('Model loaded.')

    img_name = os.path.basename(input_image)
    out_path = os.path.join(output_dir, "output_" + '.'.join(img_name.split('.')[:-1]) + '.png')
    inference_image(model, device, input_image, prompt_image, prompt_target, out_path)
    print('Finished.')
