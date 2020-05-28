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

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, TEST_EPOCH, DATASET_DIR, TEST_DIR, MODEL_DIR, ARCHITECTURE
from efficientnet_module.modules.utils import dash_path, crop_from_center_img




def main(model_dir, epoch, test_dir, classes_dir, thresholds, mode, visualization):
    """
    mode:
    1 : have labels.txt
    0 : don't have label
    """
    model = EfficientNetWrapper(INPUT_SIZE)
    model.load_classes(classes_dir)
    model.load_weight(model_dir, epoch, ARCHITECTURE, have_weight_only=False)

    images_list = []
    try:
        gt_list = []
        with open('%s/classes.txt' % classes_dir) as cl:
            classes = cl.readlines()
        list_class = [class_name.strip() for class_name in classes]
    except FileNotFoundError:
        print("Folder pointing in is a Pass folder")
        # mode = 0

    try:
        with open('%s/labels.txt' % test_dir) as f:
            read_file = f.readlines()
            # print(read_file)
            for x in read_file:
                # print(type(x.strip()))
                value = x.strip()
                if len(value) > 3:
                    images_list.append(value)
                else:
                    gt_list.append(value)
    except FileNotFoundError:
        mode = 0
        for image in glob.glob(test_dir+ dash_path +"*"):
            if imghdr.what(image) is not None:
                images_list.append(image.split(dash_path)[-1])
                # print(image.split(dash_path)[-1])
            else:
                pass
    
    detection_record = []
   
    portion_all = []
    mis_class_folder ='Mis-classified' + '/' + TEST_DIR.split(dash_path)[-1]
    escape_folder = 'Escape' + '/' + TEST_DIR.split(dash_path)[-1]
    if not os.path.exists(mis_class_folder) and not os.path.exists(escape_folder):
        os.makedirs(mis_class_folder)
        os.makedirs(escape_folder)

    score_list = []
    with tqdm.tqdm(total=len(images_list)) as pbar:
        # for idx , image_name in itertools.islice(enumerate(images_list), len(images_list)):
        for idx , image_name in itertools.islice(enumerate(images_list), max(15,len(images_list))):
            
            img = cv2.imread('%s/%s' % (test_dir, image_name))
            # print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prediction = model.predict_one(img, False, thresholds, visualize=visualization)

            for i in range(len(list_class)):
                if prediction[2] != None:
                    if prediction[2] == list_class[i]:
                        # print('Pred: %s - %s' % (i, gt_list[idx]))
                        if mode == 1:
                            record = evaluation.get_record_from_txt_and_result(int(gt_list[idx]) + 1,int(i) + 1, prediction[1])
                            if int(gt_list[idx]) !=  int(i):
                                # pass
                                cv2.imwrite(mis_class_folder + dash_path + image_name+ f"_Sc{round(prediction[1],2)}_Tr{list_class[int(gt_list[idx])]}_Pr{prediction[2]}.bmp", img)
                            else:
                                pass
                        else:
                            record = evaluation.get_record_from_result_only(int(i) + 1, prediction[1])
                else:
                    record = evaluation.get_record_from_txt_and_result(int(gt_list[idx]) + 1, [])
                    cv2.imwrite(escape_folder + dash_path + image_name, img)

            score_prediction = model.predict_one(img, True, thresholds, visualize=False)

            image_result = [image_name]
            image_result.extend([round(x,2) for x in score_prediction[0]])
            # print(gt_list)
            if mode == 1:
                image_result.extend(gt_list[idx])
            else:
                pass

            score_list.append(image_result)
            # print(record)
            detection_record.append(record)
            # print(detection_record)
            pbar.update()

    # print(detection_record)
    evaluation.evaluate_img_classification(detection_record, classes_dir, mode)
    header =["image_name"]
    header.extend(list_class)
    if mode == 1:
        header.append("Groundtruth")
    else:
        pass
    with open(TEST_DIR.split(dash_path)[-1]+".csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(score_list)

if __name__ == '__main__':
    classes_dir = MODEL_DIR
    # mode = 1 - have label . mode = 0 don't have label
    mode = 1
    thresholds = None
    main(MODEL_DIR, TEST_EPOCH, TEST_DIR, classes_dir, thresholds, mode, False)
