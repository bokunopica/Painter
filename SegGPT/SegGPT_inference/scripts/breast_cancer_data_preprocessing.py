import os
from tqdm import trange
from PIL import Image


def union_mask(image_array):
    if len(image_array) == 0:
        return image_array[0]
    base_image = image_array[0]
    for image in image_array[1:]:
        base_image.paste(image, (0,0), mask = image)
    return base_image

def get_image_id(image_name):
    return image_name.split('(')[1].split(')')[0]

def get_mask_id(image_name):
    split_list = image_name.split('mask')
    mask_id_str = split_list[1]
    if len(mask_id_str) == 0:
        return 0
    else:
        mask_id_str = mask_id_str.replace('_', '')
        return int(mask_id_str)
    
def main():
    data_folder = "/mnt/c/data/breastCancer/Dataset_BUSI_with_GT"
    save_path = "/mnt/c/data/breastCancer/processed"
    class_list = ["benign", "malignant", "normal"]

    for _class in class_list:
        image_name_list = os.listdir(f"{data_folder}/{_class}")
        for i in trange(len(image_name_list)):
            image_name = image_name_list[i]
            image_name = image_name.split('.')[0]
            if 'mask' in image_name:
                continue
            image_id = int(get_image_id(image_name))
            

            # 获取相应的mask 有多个的话合并
            mask_file_name_list = filter(
                lambda x: (
                    'mask' in x
                    and image_name in x
                ),
                image_name_list,
            )
            

            mask_image_list = []
            for mask_file_name in mask_file_name_list:
                mask_image = Image.open(f"{data_folder}/{_class}/{mask_file_name}")
                mask_image_list.append(mask_image)
            
            union_mask_image = union_mask(mask_image_list)
            source_image = Image.open(f"{data_folder}/{_class}/{image_name}.png")

            # 储存图像和对应的合并mask
            source_image.save(f"{save_path}/source/{_class}_{'%03d'%image_id}.png")
            union_mask_image.save(f"{save_path}/target/{_class}_{'%03d'%image_id}_mask.png")


if __name__ == "__main__":
    main()