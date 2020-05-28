import os
import sys
import tqdm
import cv2
import itertools
import evaluation_classify as evaluation
import imghdr
import glob
import numpy as np
import csv
import pandas as pd

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config_kaggle import INPUT_SIZE, TEST_EPOCH, DATASET_DIR, TEST_DIR, MODEL_DIR, ARCHITECTURE
from efficientnet_module.modules.utils import dash_path, crop_from_center_img




def main(dataset_dir, model_dir, epoch, test_dir, thresholds, visualization):
    """
    mode:
    1 : have labels.txt
    0 : don't have label
    """
    # model = EfficientNetWrapper(INPUT_SIZE)
    # model.load_classes_kaggle(classes)
    # # model.load_classes(classes_dir)
    # model.load_weight(model_dir, epoch, ARCHITECTURE, have_weight_only=False)

    d_train = pd.read_csv(DATASET_DIR + "train.csv")
    classes = d_train.columns.values[1:]
    d_test = pd.read_csv(DATASET_DIR + "test.csv")
    test_ids = d_test.values

    # print(test_ids)
    # print(classes)
    list_class = classes

    images_list =[x[0] for x in test_ids]
    score_list = []
    with tqdm.tqdm(total=len(images_list)) as pbar:
        # for idx , image_name in itertools.islice(enumerate(images_list), len(images_list)):
        for idx , image_name in itertools.islice(enumerate(images_list), max(15,len(images_list))):
            
            img = cv2.imread('%s/%s.jpg' % (test_dir, image_name))
            # print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            score_prediction = model.predict_one(img, True, thresholds, visualize=False)

            image_result = [image_name]
            image_result.extend([round(x,2) for x in score_prediction[0]])
          
            score_list.append(image_result)
           
            detection_record.append(record)
            
            pbar.update()

    # print(detection_record)
    evaluation.evaluate_img_classification(detection_record, classes_dir, mode)
    header =["image_id"]
    header.extend(list_class)

    with open(TEST_DIR.split(dash_path)[-1]+".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(score_list)

if __name__ == '__main__':
    classes_dir = MODEL_DIR
    # mode = 1 - have label . mode = 0 don't have label
    mode = 1
    thresholds = None
    
    main(DATASET_DIR ,MODEL_DIR, TEST_EPOCH, TEST_DIR, thresholds, False)
