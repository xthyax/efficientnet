import os
import sys
import tqdm
import cv2
import itertools
import evaluation_classify as evaluation
import imghdr
import glob
import numpy as np
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, TEST_EPOCH, DATASET_DIR, TEST_DIR, MODEL_DIR, ARCHITECTURE
from efficientnet_module.modules.utils import dash_path, crop_from_center_img


rotate_option_list = [0]
def result_visualize(portion_scale, img, model, list_class, thresholds, rotate_image_option,return_visualize):
    score_all = []
    class_all = []
    id_class_all = []
    if img.shape[0] >500 :
        portion_scale = 2
        image_h, image_w, _ = img.shape
        portion_h = image_h / portion_scale
        portion_w = image_w / portion_scale
        
        # augmentation = iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
        sub_score_all = []
        for i in range(portion_scale):
            for j in range(portion_scale):
                # top bottom, left right
                image_portion = img[int(portion_h*i):int(portion_h*(i+1)), int(portion_w*j): int(portion_w*(j+1))]
                for rotate in rotate_option_list:
                    image_predict = rotate_image_option.get(rotate,'')(image_portion) if rotate != 0 else image_portion

                    prediction = model.predict_one(image_predict, True, thresholds)
                    
                    prediction = prediction.tolist()[0]
                    append_score =[x if prediction.index(x) != len(prediction) -1 else x/2  for x in prediction ]
                    sub_score_all.append(append_score)
                    # sub_class_all.append(prediction[2])
                    # sub_id_class_all.append(prediction[0])

                # print(sub_score_all)
                # print(sub_class_all)
                # print(sub_id_class_all)
                
                # if  0 < len(list(dict.fromkeys(sub_id_class_all))) < 2:
                #     keep = np.array(sub_score_all) == max(sub_score_all)
                #     portion_keep = np.nonzero(keep.astype('uint8'))[0]
                #     # index_keep = np.nonzero(keep)
                #     score_ = max(sub_score_all)
                #     class_ = list(dict.fromkeys(sub_class_all))[0]
                #     id_ = list(dict.fromkeys(sub_id_class_all))[0]
                # else:
                #     print(sub_score_all)
                #     print(sub_class_all)
                #     print(sub_id_class_all)
                #     keep = np.array(sub_score_all) > 0.8
                #     portion_keep = np.nonzero(keep.astype('uint8'))[0]
                #     # print("This portion image don't have enough evidence to prove it contain a defect")
                #     # continue
                #     try:
                #         score_ = max([sub_score_all.index(x) for x in portion_keep])
                #         class_keep = sub_class_all.index(sub_class_all.index(score_))
                #         class_ = sub_class_all[class_keep]
                #         id_ = sub_id_class_all[class_keep]
                #         # pass
                #     except ValueError:
                #         print("This portion image don't have enough evidence to prove it contain a defect")
                #         continue
               
                # score_all.append(score_)
                # class_all.append(class_)
                # id_class_all.append(id_)
        
        sub_score_all = np.array(sub_score_all)
        print(sub_score_all)
        print(sub_score_all.shape)
        average_score = np.mean(sub_score_all,axis=0)
        print(average_score)
        score_all.append(average_score.tolist())
        print(score_all)     

        sub_score_all = []      
        for w, h in [[384,288], [img.shape[1], img.shape[0]]]:
            # print (w, h)
            image = crop_from_center_img(img, w, h)
            
            sub_class_all = []
            sub_id_class_all = []   
            if w < 500:
                for rotate in rotate_option_list:
                    image_predict = rotate_image_option.get(rotate,'')(image) if rotate != 0 else image

                    prediction = model.predict_one(image_predict, True, thresholds)

                    prediction = prediction.tolist()[0]
                    append_score =[x if prediction.index(x) != len(prediction) -1 else x/2  for x in prediction ]
                    sub_score_all.append(append_score)
                    # np.append(score_all, prediction)
                    # sub_class_all.append(prediction[2])
                    # sub_id_class_all.append(prediction[0])

                # if  0 < len(list(dict.fromkeys(sub_id_class_all))) < 2:
                #     keep = np.array(sub_score_all) == max(sub_score_all)
                #     portion_keep = np.nonzero(keep.astype('uint8'))[0]
                #     score_ = max(sub_score_all)
                #     class_ = list(dict.fromkeys(sub_class_all))[0]
                #     id_ = list(dict.fromkeys(sub_id_class_all))[0]
                # else:
                #     print(sub_score_all)
                #     print(sub_class_all)
                #     print(sub_id_class_all)
                #     keep = np.array(sub_score_all) > 0.8
                #     portion_keep = np.nonzero(keep.astype('uint8'))[0]
                #     # print("This portion image don't have enough evidence to prove it contain a defect")
                #     # continue
                #     try:
                #         score_ = max([sub_score_all.index(x) for x in portion_keep])
                #         class_keep = sub_class_all.index(sub_class_all.index(score_))
                #         class_ = sub_class_all[class_keep]
                #         id_ = sub_id_class_all[class_keep]
                #         # pass
                #     except ValueError:
                #         print("This portion image don't have enough evidence to prove it contain a defect")
                #         continue

                # score_all.append(score_)
                # class_all.append(class_)
                # id_class_all.append(id_)
            # prediction = model.predict_one(image, thresholds,  visualize=visualization)
            # print('GT: %s' % list_class[int(gt_list[idx])] if len(gt_list) != 0 else "Pass")
            # print('Pred: %s - %s' % (prediction[2], prediction[1]))
            # print(prediction[0])
            else:
                prediction = model.predict_one(img, True, thresholds)
                # np.append(score_all, prediction)
                prediction = prediction.tolist()[0]
                append_score =[x if prediction.index(x) != len(prediction) -1 else x/2  for x in prediction ]
                sub_score_all.append(append_score)
                
        print(sub_score_all)
        sub_score_all = np.array(sub_score_all)
        average_score = np.mean(sub_score_all,axis=0)
        score_all.append(average_score.tolist())

        print("="*40)
        print(score_all)
        print("="*40)
        score_all = np.mean(np.array(score_all), axis=0).tolist()
        last_score = max(score_all)
        last_class = list_class[score_all.index(last_score)]
        last_id_class =  score_all.index(last_score)
    else:
        prediction = model.predict_one(img, True, thresholds)
        last_score = max(prediction)
        last_class = list_class[prediction.index(last_score)]
        last_id_class = prediction.index(last_score)

    if return_visualize:
        cv2.namedWindow(f'{last_class} - {last_score}', cv2.WINDOW_NORMAL) 
        cv2.resizeWindow(f'{last_class} - {last_score}', 600, 600)
        cv2.imshow(f'{last_class} - {last_score}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return last_score, last_class, last_id_class

def main(model_dir, epoch, test_dir, classes_dir, thresholds, mode, visualization):
    """
    mode:
    1 : have labels.txt
    0 : don't have label
    """
    rotate_image_option = {
        1: lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
        2: lambda x: cv2.rotate(x, cv2.ROTATE_180),
        3: lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE)
    }
    model = EfficientNetWrapper(INPUT_SIZE)
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
    with tqdm.tqdm(total=len(images_list)) as pbar:
        # for idx , image_name in itertools.islice(enumerate(images_list), len(images_list)):
        for idx , image_name in itertools.islice(enumerate(images_list), max(15,len(images_list))):
            
            img = cv2.imread('%s/%s' % (test_dir, image_name))
            # print(img.shape)
            last_score, last_class, last_id_class = result_visualize(2, img, model, list_class, thresholds, rotate_image_option, visualization)

            for i in range(len(list_class)):
                if last_class == list_class[i]:
                    # print('Pred: %s - %s' % (i, gt_list[idx]))
                    if mode == 1:
                        record = evaluation.get_record_from_txt_and_result(int(gt_list[idx]),int(i), prediction[2])
                    else:
                        record = evaluation.get_record_from_result_only(int(i), last_score)

            # print(record)
            detection_record.append(record)
            # print(detection_record)
            pbar.update()
    print(detection_record)
    evaluation.evaluate_img_classification(detection_record, classes_dir, mode)

if __name__ == '__main__':
    classes_dir = MODEL_DIR
    # mode = 1 - have label . mode = 0 don't have label
    mode = 1
    thresholds = [0] * 14 
    thresholds.append(0.9)
    main(MODEL_DIR, TEST_EPOCH, TEST_DIR, classes_dir, thresholds, mode, True)
