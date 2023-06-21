import os
import numpy as np
from PIL import Image
from tqdm import trange


def reverse_color(img: Image) -> Image:
    matrix = 255 - np.asarray(img)  # 图像转矩阵 并反色
    new_img = Image.fromarray(matrix)  # 矩阵转图像
    return new_img


def binary_dice(s, g):
    """
    calculate the Dice score of two N-d volumes.
    s: the segmentation volume of numpy array
    g: the ground truth volume of numpy array
    """
    assert len(s.shape), len(g.shape)
    prod = np.multiply(s, g)
    s0 = prod.sum()
    dice = (2.0 * s0 + 1e-10) / (s.sum() + g.sum() + 1e-10)
    return dice


def dice_evaluation(ground_truth_folder, result_folder, need_reverse_color=False):
    dir_list = os.listdir(result_folder)
    cnt = 0
    total_dice = 0
    # for i in trange(len(dir_list)):
    for i in trange(len(dir_list)):
        item = dir_list[i]
        if not item.endswith("mask.png"):
            continue
        g_img = Image.open(f"{ground_truth_folder}/{item.replace('output_', '')}").convert('RGB')
        if need_reverse_color:
            g_img = reverse_color(g_img)
        s_img = Image.open(f"{result_folder}/{item}").convert('RGB')
        dice = binary_dice(
            s=np.asarray(s_img.convert('1')),
            g=np.asarray(g_img.convert('1')),
        )
        cnt+=1
        total_dice += dice
    print(total_dice/cnt)


if __name__ == "__main__":
    dice_evaluation(
        ground_truth_folder = "/run/media/breastCancer/processed/target",
        result_folder = "/run/media/breastCancer/results_seggpt_origin",
        need_reverse_color=True
    )

    dice_evaluation(
        ground_truth_folder = "/run/media/breastCancer/processed/target",
        result_folder = "/run/media/breastCancer/results_seggpt_finetune_30",
        need_reverse_color=False
    )

    dice_evaluation(
        ground_truth_folder = "/run/media/breastCancer/processed/target",
        result_folder = "/run/media/breastCancer/results_seggpt_finetune_300",
        need_reverse_color=False
    )
    
    
