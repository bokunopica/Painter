from PIL import Image
import numpy as np



def reverse_color(img: Image) -> Image:
    matrix = 255 - np.asarray(img)  # 图像转矩阵 并反色
    new_img = Image.fromarray(matrix)  # 矩阵转图像
    return new_img