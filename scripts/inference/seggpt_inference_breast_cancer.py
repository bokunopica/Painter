import os
import sys
import json
import argparse
import random
import torch
import numpy as np
from tqdm import trange

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__),
            ),
        )
    )
)

from SegGPT.SegGPT_inference.seggpt_inference import inference_image
from SegGPT.SegGPT_train import models_seggpt


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser("SegGPT inference", add_help=False)
    parser.add_argument(
        "--ckpt_path", type=str, help="path to ckpt", default="seggpt_vit_large.pth"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="dir to ckpt",
        default="seggpt_vit_large_patch16_input896x448",
    )
    parser.add_argument(
        "--input_image", type=str, help="path to input image to be tested", default=None
    )
    parser.add_argument(
        "--input_video", type=str, help="path to input video to be tested", default=None
    )
    parser.add_argument(
        "--num_frames", type=int, help="number of prompt frames in video", default=0
    )
    parser.add_argument(
        "--prompt_image", type=str, nargs="+", help="path to prompt image", default=None
    )
    parser.add_argument(
        "--prompt_target",
        type=str,
        nargs="+",
        help="path to prompt target",
        default=None,
    )
    parser.add_argument(
        "--seg_type",
        type=str,
        help="embedding for segmentation types",
        choices=["instance", "semantic"],
        default="instance",
    )
    parser.add_argument("--device", type=str, help="cuda or cpu", default="cuda")
    parser.add_argument("--output_dir", type=str, help="path to output", default="./")
    return parser.parse_args()


def prepare_model(
    chkpt_dir, arch="seggpt_vit_large_patch16_input896x448", seg_type="instance"
):
    # build model
    model = getattr(models_seggpt, arch)()
    model.seg_type = seg_type
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    model.eval()
    return model


def split_train_test(source_list, percentage):
    total_len = len(source_list)
    cut_len = int(percentage * total_len)
    return source_list[:cut_len], source_list[cut_len:]


def generate_outputs(model_path, save_dir, device, reverse_mask_color=False):
    ckpt_path = model_path
    model = "seggpt_vit_large_patch16_input896x448"
    seg_type = "instance"
    output_dir = save_dir

    train_meta_dir = "/run/media/breastCancer/processed/meta_train.json"
    test_meta_dir = "/run/media/breastCancer/processed/meta_test.json"

    # prompt inference split
    with open(train_meta_dir, 'r') as f:
        prompt_list = json.loads(f.read())

    with open(test_meta_dir, 'r') as f:
        inference_list = json.loads(f.read())

    benign_list = []
    malignant_list = []
    normal_list = []
    for item in prompt_list:
        if "benign" == item['_class']:
            benign_list.append(item)
        elif "malignant" == item['_class']:
            malignant_list.append(item)
        else:
            normal_list.append(item)

    percentage = 0.025  # benign=8 malignant=4 normal=2
    random.seed(111)
    random.shuffle(benign_list)
    random.shuffle(malignant_list)
    random.shuffle(normal_list)

    # benign_list, _ = split_train_test(benign_list, 0.05)
    # malignant_list, _ = split_train_test(malignant_list, 0.05)
    # normal_list, _ = split_train_test(normal_list, 0.05)

    prompt_list = []
    inference_list = []

    for _list in [benign_list, malignant_list, normal_list]:
        _train, _test = split_train_test(_list, percentage)
        print(f"len={len(_train)}")
        prompt_list += _train
        inference_list += _test

    input_image_list = []
    prompt_image = []
    prompt_target = []

    for item in prompt_list:
        prompt_image.append(item['image_path'])
        prompt_target.append(item['target_path'])

    for item in inference_list:
        input_image_list.append(item['image_path'])

    print(len(prompt_list))
    print(len(inference_list))

    model = prepare_model(ckpt_path, model, seg_type).to(device)
    print("Model loaded.")

    for i in trange(len(input_image_list)):
        input_image = input_image_list[i]
        img_name = os.path.basename(input_image)
        out_path = os.path.join(
            output_dir, "output_" + ".".join(img_name.split(".")[:-1]) + ".png"
        )
        inference_image(
            model,
            device,
            input_image,
            prompt_image,
            prompt_target,
            out_path,
            reverse_mask_color=reverse_mask_color,
        )
    print("Finished.")


if __name__ == "__main__":
    # origin model
    # generate_outputs(
    #     model_path="/home/qianq/mycodes/Painter/SegGPT/SegGPT_inference/pretrained_seggpt/seggpt_vit_large.pth",
    #     save_dir="/run/media/breastCancer/results_seggpt_origin",
    #     device="cuda:0",
    #     reverse_mask_color=True
    # )

    # finetuned model - 30 epochs
    # generate_outputs(
    #     model_path="/home/qianq/mycodes/Painter/output_dir/seggpt-finetune-breastCancer-01/checkpoint-29.pth",
    #     save_dir="/run/media/breastCancer/results_seggpt_finetune_30",
    #     device="cuda:1",
    # )

    # finetuned model - 300 epochs
    generate_outputs(
        model_path="/home/qianq/mycodes/Painter/output_dir/seggpt-finetune-breastCancer-02/checkpoint-299.pth",
        save_dir="/run/media/breastCancer/results_seggpt_finetune_300",
        device="cuda:0"
    )
    pass
