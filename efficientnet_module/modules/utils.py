import fnmatch
import os
import glob
from sklearn.utils import class_weight
from prettytable import PrettyTable
import json
import cv2
# from PIL import Image

import numpy as np
# from .config import INPUT_SIZE, ARCHITECTURE, BATCH, DATASET_DIR, Optimizer_
dash_path = "\\" if os.name =="nt" else "/"

def to_onehot(labels, num_of_classes):
    if type(labels) is list:
        labels = [int(label) for label in labels]
        arr = np.array(labels, dtype=np.int)
        onehot = np.zeros((arr.size, num_of_classes))
        onehot[np.arange(arr.size), arr] = 1

    else:
        onehot = np.zeros((num_of_classes,), dtype=np.int)
        onehot[int(labels)] = 1

    return onehot


def multi_threshold(Y, thresholds):
    if Y.shape[-1] != len(thresholds):
        raise ValueError('Mismatching thresholds and output classes')

    thresholds = np.array(thresholds)
    thresholds = thresholds.reshape((1, thresholds.shape[0]))
    keep = Y > thresholds
    score = keep * Y
    class_id = np.argmax(score, axis=-1)
    class_score = np.max(score, axis=-1)
    if class_score == 0:
        return None
    return class_id, class_score


def recursive_glob(treeroot, pattern):
    results = []
    for base, dirs, files in os.walk(treeroot):
        goodfiles = fnmatch.filter(files, pattern)
        results.extend(os.path.join(base, f) for f in goodfiles)
    return results

def recursive_folder(parent_folder):
    # TODO: make the search more deep
    # Sallow search
    sub_folder = next(os.walk(parent_folder))[1]
    list_subFolder = []
    if len(sub_folder) != 0 :
        for folder in sub_folder:
            sub_parentFolder = os.path.join(parent_folder, folder)
            # print(sub_parentFolder)
            list_subFolder.append(sub_parentFolder)
        return list_subFolder
    else:
        return parent_folder
    # print(parent_folder)
    # print(sub_folder)

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    return "".join([x if x.isalnum() else "_" for x in value])

def compute_class_weight(class_count):
    y = [[i] * v for i, v in enumerate(class_count)]
    y = flatten_list(y)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weights

def flatten_list(list_):
    return [item for sublist in list_ for item in sublist]

def metadata_count(dir, classes, gt_list, show_table=True):
    test_dir = dir if isinstance(dir,list) else [dir]
    class_list = list(dict.fromkeys(gt_list))

    Table = PrettyTable()
    Table.field_names = ['Defect', 'Number of images']
    count_class = [0] * len(class_list)
    for i in range(len(gt_list)):
        for j in range(len(class_list)):
            if gt_list[i] == class_list[j]:
                count_class[j] += 1

    metadata = {}
    # Empty folder check
    if len(gt_list) != 0: 
        for i in range(len(classes)):
            metadata.update({classes[i]: count_class[i]})
            Table.add_row([classes[i],count_class[i]])
    if show_table:
        print(f"[DEBUG] Path: {test_dir}")
        print(Table)
    if any('train' in sub_testdir.lower() for sub_testdir in test_dir):
        # print("[DEBUG] Had return metadata value")
        return metadata

def config_dump(save_path, config):
    data ={
        "LEARNING_RATE":f"{config.LEARNING_RATE}",
        "INPUT_SIZE": f"{config.INPUT_SIZE}",
        "BATCH": f"{config.BATCH_SIZE}",
        "EPOCH": f"{config.EPOCH}",
        "ARCHITECTURE": f"{config.ARCHITECTURE}",
        "OPTIMIZER": f"{config.OPTIMIZER}",
        "DATASET_PATH": f"{config.DATASET_PATH}"
    }
    with open(save_path + "/config.json", "w") as outfile:
        json.dump(data, outfile)

def load_and_crop(image_path, input_size=0):
    """ Load image and return image with specific crop size

    Input:
        image_path : Ex:Dataset/Train/img01.bmp
        input_size : any specific size
        
    Output:
        image after crop
    """
    image = cv2.imread(image_path)
    json_path = image_path + ".json"
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    size_image = image.shape

    try :
        with open(json_path, encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            box = json_data['box']
            center_x = box['centerX'][0]
            center_y = box['centerY'][0]
            widthBox = box['widthBox'][0]
            heightBox = box['heightBox'][0]
            class_gt = json_data['classId'][0]
    except:
        print(f"Can't find {json_path}")
        # Crop center image if no json found
        center_x = int(size_image[1] / 2)
        center_y = int(size_image[0] / 2)

    new_w = max(input_size, widthBox)
    new_h = max(input_size, heightBox)

    left, right = center_x - new_w / 2, center_x + new_w / 2
    top, bottom = center_y - new_h / 2, center_y + new_h / 2

    left, top = round(max(0, left)), round(max(0, top))
    right, bottom = round(min(size_image[1] - 0, right)), round(min(size_image[0] - 0, bottom))

    return image[int(top):int(bottom), int(left):int(right)], class_gt