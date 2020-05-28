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
import json

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, TEST_EPOCH, DATASET_DIR, TEST_DIR, MODEL_DIR, ARCHITECTURE
from efficientnet_module.modules.utils import dash_path, crop_off_image_classification, crop_image_with_json




def main(model_dir, epoch, test_dir, classes_dir, thresholds, mode, visualization):
    """
    mode:
    1 : have labels.txt
    0 : don't have label
    """
    model = EfficientNetWrapper(INPUT_SIZE)
    model.load_classes_KLA(classes_dir)
    model.load_weight(model_dir, epoch, ARCHITECTURE, have_weight_only=False)

    images_list = []
    try:
        with open('%s/config_KLA.json' % classes_dir) as f:
            data = json.load(f)
        list_class = data["ClassName"]
    except FileNotFoundError:
        print("Folder pointing in is a Pass folder")
        # mode = 0

    for image in glob.glob(test_dir+ dash_path +"*"):
        if imghdr.what(image) is not None:
            images_list.append(image.split(dash_path)[-1])
            # print(image.split(dash_path)[-1])
        else:
            pass
    
    detection_record = []
    mis_class_folder_OK = '1.Mis-classified' + '/' + TEST_DIR.split(dash_path)[-1] + "/" + "OK"
    mis_class_folder_UK = '1.Mis-classified' + '/' + TEST_DIR.split(dash_path)[-1] + "/" + "UK"
    reject_class_folder = '2.Result_output' + '/' + TEST_DIR.split(dash_path)[-1] + "/" + "Reject"
    pass_class_folder =  '2.Result_output' + '/' + TEST_DIR.split(dash_path)[-1] + "/" + "Pass"
    escape_folder = '3.Escape' + '/' + TEST_DIR.split(dash_path)[-1]
    if not os.path.exists(mis_class_folder_OK) and not os.path.exists(escape_folder) and not os.path.exists(mis_class_folder_UK)\
        and not os.path.exists(reject_class_folder) and not os.path.exists(pass_class_folder) :
        os.makedirs(mis_class_folder_OK)
        os.makedirs(mis_class_folder_UK)
        os.makedirs(reject_class_folder)
        os.makedirs(pass_class_folder)
        os.makedirs(escape_folder)

    score_list = []
    with tqdm.tqdm(total=len(images_list)) as pbar:
        # for idx , image_name in itertools.islice(enumerate(images_list), len(images_list)):
        for idx , image_name in itertools.islice(enumerate(images_list), max(15,len(images_list))):
            json_path = f'{test_dir}/{image_name}'+ ".json"

            with open(json_path) as lb:
                label = json.load(lb)
            center_point = [label['box']['centerX'][0], label['box']['centerY'][0]]
            gt_ = label['classId'][0]

            img_path = '%s/%s' % (test_dir, image_name)

            img = crop_image_with_json(json_path, img_path , INPUT_SIZE)

            prediction = model.predict_one(img, False, thresholds, visualize=visualization)
            # print(image_name,'-',f"[{prediction[2]}-{prediction[1]}]")
            for i in range(len(list_class)):
                if prediction[2] != None:
                    if prediction[2] == list_class[i]:
                        # print('Pred: %s - %s' % (i, gt_list[idx]))
                        # print(f'{prediction[2]} - {list_class.index(gt_)}')
                        if mode == 1:
                            record = evaluation.get_record_from_txt_and_result(list_class.index(gt_) + 1,int(i) + 1, prediction[1])
                            if int(i) == 1 and list_class.index(gt_) == 0: # DT : Pass -- GT: Reject  :: UK
                                # pass
                                cv2.imwrite(mis_class_folder_UK + dash_path + image_name, img)

                            elif int(i) == 0 and list_class.index(gt_) == 1: # DT: Reject -- GT: Pass :: OK
                                cv2.imwrite(mis_class_folder_OK + dash_path + image_name, img)

                            elif int(i) == 0 and list_class.index(gt_) == 0: # DT: Reject -- GT: Reject :: Reject
                                cv2.imwrite(reject_class_folder + dash_path + image_name, img)

                            elif int(i) == 1 and list_class.index(gt_) == 1: # DT: Pass -- GT: Pass :: Pass
                                cv2.imwrite(pass_class_folder + dash_path + image_name, img)

                            else:
                                pass
                        else:
                            record = evaluation.get_record_from_result_only(int(i) + 1, prediction[1])
                else:
                    record = evaluation.get_record_from_txt_and_result(list_class.index(gt_) + 1, [])
                    cv2.imwrite(escape_folder + dash_path + image_name, img)

            # score_prediction = model.predict_one(img, True, thresholds, visualize=False)

            # image_result = [image_name]
            # image_result.extend([round(x,2) for x in score_prediction[0]])
            # print(gt_list)
            # if mode == 1:
            #     image_result.extend(gt_list[idx])
            # else:
            #     pass

            # score_list.append(image_result)
            # print(record)
            detection_record.append(record)
            # print(detection_record)
            pbar.update()

    # print(detection_record)
    evaluation.evaluate_img_classification(detection_record, classes_dir, mode)
    # header =["image_name"]
    # header.extend(list_class)
    # if mode == 1:
    #     header.append("Groundtruth")
    # else:
    #     pass
    # with open(TEST_DIR.split(dash_path)[-1]+".csv", "w", newline='') as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(header)
    #     writer.writerows(score_list)

if __name__ == '__main__':
    classes_dir = DATASET_DIR
    # mode = 1 - have label . mode = 0 don't have label
    mode = 1
    thresholds = None
    main(MODEL_DIR, TEST_EPOCH, TEST_DIR, classes_dir, thresholds, mode, False)
