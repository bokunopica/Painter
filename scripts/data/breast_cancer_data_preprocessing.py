import os
import random
import json
from tqdm import trange
from PIL import Image


def union_mask(image_array):
    if len(image_array) == 0:
        return image_array[0]
    base_image = image_array[0]
    for image in image_array[1:]:
        base_image.paste(image, (0, 0), mask=image)
    return base_image


def get_image_id(image_name):
    return image_name.split("(")[1].split(")")[0]


def get_mask_id(image_name):
    split_list = image_name.split("mask")
    mask_id_str = split_list[1]
    if len(mask_id_str) == 0:
        return 0
    else:
        mask_id_str = mask_id_str.replace("_", "")
        return int(mask_id_str)


def build_breast_cancer_data_json():
    source_base_dir = "/mnt/c/data/breastCancer/processed/source"
    target_base_dir = "/mnt/c/data/breastCancer/processed/target"
    json_path = "/mnt/c/data/breastCancer/processed/meta.json"
    # type_list = ['benign', 'normal', 'malignant']
    source_filename_list = os.listdir(source_base_dir)
    with open(json_path, "w") as f:
        meta = []
        for source in source_filename_list:
            target = source.replace(".png", "_mask.png")
            single_meta = {
                "image_path": f"{source_base_dir}/{source}",
                "target_path": f"{target_base_dir}/{target}",
                "type": "instance",
            }
            meta.append(single_meta)
        f.write(json.dumps(meta))


def split_breast_cancer_data_json(percentage=0.8):
    json_path = "/run/media/breastCancer/processed/meta.json"
    train_json_path = "/run/media/breastCancer/processed/meta_train.json"
    test_json_path = "/run/media/breastCancer/processed/meta_test.json"
    with open(json_path, "r") as f:
        data_list = json.loads(f.read())

    def get_class(image_path):
        _class_list = ["benign", "malignant", "normal"]
        for _class in _class_list:
            if _class in image_path:
                return _class
        return None

    def split_single_train_test(single_data_source_list, percentage):
        total_len = len(single_data_source_list)
        cut_len = int(percentage * total_len)
        return single_data_source_list[:cut_len], single_data_source_list[cut_len:]

    def split_train_test(source_list, percentage):
        train_list = []
        test_list = []
        for single_data_source_list in source_list:
            _train_list, _test_list = split_single_train_test(
                single_data_source_list, percentage
            )
            train_list += _train_list
            test_list += _test_list
        return train_list, test_list

    benign_list = []
    malignant_list = []
    normal_list = []

    for i in range(len(data_list)):
        data = data_list[i]
        _class = get_class(data["image_path"])
        data["_class"] = _class
        if "benign" == _class:
            benign_list.append(data)
        elif "malignant" == _class:
            malignant_list.append(data)
        else:
            normal_list.append(data)

    random.seed(111)
    random.shuffle(benign_list)
    random.shuffle(malignant_list)
    random.shuffle(normal_list)

    train_list, test_list = split_train_test(
        [benign_list, malignant_list, normal_list], percentage
    )
    with open(train_json_path, "w") as f:
        f.write(json.dumps([data for data in train_list]))
    with open(test_json_path, "w") as f:
        f.write(json.dumps([data for data in test_list]))


def main():
    data_folder = "/mnt/c/data/breastCancer/Dataset_BUSI_with_GT"
    save_path = "/mnt/c/data/breastCancer/processed"
    class_list = ["benign", "malignant", "normal"]

    for _class in class_list:
        image_name_list = os.listdir(f"{data_folder}/{_class}")
        for i in trange(len(image_name_list)):
            image_name = image_name_list[i]
            image_name = image_name.split(".")[0]
            if "mask" in image_name:
                continue
            image_id = int(get_image_id(image_name))

            # 获取相应的mask 有多个的话合并
            mask_file_name_list = filter(
                lambda x: ("mask" in x and image_name in x),
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
            union_mask_image.save(
                f"{save_path}/target/{_class}_{'%03d'%image_id}_mask.png"
            )




if __name__ == "__main__":
    main()
