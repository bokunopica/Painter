import os
import numpy as np
from PIL import Image
from tqdm import trange
from util.image_utils import reverse_color


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


if __name__ == "__main__":
    ground_truth_folder = "/mnt/c/data/breastCancer/processed/target"
    result_folder = "/mnt/c/data/breastCancer/results"
    dir_list = os.listdir(result_folder)
    cnt = 0
    total_dice = 0
    # for i in trange(len(dir_list)):
    for i in trange(len(dir_list)):
        item = dir_list[i]
        if not item.endswith("mask.png"):
            continue
        g_img = Image.open(f"{ground_truth_folder}/{item.replace('output_', '')}").convert('RGB')
        s_img = Image.open(f"{result_folder}/{item}").convert('RGB')
        dice = binary_dice(
            s=np.asarray(s_img.convert('1')),
            g=np.asarray(reverse_color(g_img).convert('1')),
        )
        cnt+=1
        total_dice += dice
    print(total_dice/cnt)
