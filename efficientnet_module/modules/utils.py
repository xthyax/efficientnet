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

def metadata_count(dir, classes, show_table=True):
    gt_list = []
    test_dir = dir if isinstance(dir,list) else [dir]
    try:
        for subdir in test_dir:
            # print(f"[DEBUG] Path: {subdir}")
            for json_file in glob.glob(subdir + "/*.json"):
                with open(json_file) as f:
                    obj = json.load(f)
                value = obj['classId']
                gt_list.append(value[0])

        class_list = list(dict.fromkeys(gt_list))
        # print(class_list)

        Table = PrettyTable()
        Table.field_names = ['Defect', 'Number of images']
        count_class = [0] * len(class_list)
        for i in range(len(gt_list)):
            for j in range(len(class_list)):
                if gt_list[i] == class_list[j]:
                    count_class[j] += 1

        metadata = {}
        for i in range(len(classes)):
            metadata.update({classes[i]: count_class[i]})
            Table.add_row([classes[i],count_class[i]])
        # print(count_class)
        # print(metadata)
        if show_table:
            print(Table)
        if any('train' in sub_testdir.lower() for sub_testdir in test_dir):
        # if 'train' in sub_testdir.lower() for sub_testdir in test_dir:
            # with open (dash_path.join(test_dir.split(dash_path)[:-1]) + dash_path + 'metadata.json', 'w') as output_file:
            #     json.dump(metadata , output_file)
            return metadata
    except (FileNotFoundError, UnboundLocalError):
        print("="*50)
        print(f"Didn't find json file at #{subdir}#")
        print(f"Please re check your {json_file} file")
        print("="*50)

def crop_off_image_classification(imgpath, center_point, boxSize):
    # print(imgpath)
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # center_point = str.format("{} {}", int( img.shape[0]/2), int( img.shape[1]/2))
    # print(center_point)
    if boxSize != 0:
        # print(img)
        startX = int(center_point[0] - boxSize/2)
        startY = int(center_point[1] - boxSize/2)
        return img[startY: startY + boxSize, startX: startX + boxSize], center_point[0]-startX, center_point[1]-startY
    else:
        startX = 0
        startY = 0
        return img[startY: startY + img.shape[0], startX: startX + img.shape[1]], img.shape[0]/2, img.shape[1]/2

def image_read(image_path , size):
    desired_size = size
    im_pth = image_path

    im = cv2.imread(im_pth)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

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

    return image[int(top):int(bottom), int(left):int(right)]