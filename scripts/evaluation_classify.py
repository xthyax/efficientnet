from collections import Counter
import json
import numpy as np
from loguru import logger
# import loadJSON_label

def compute_iou(box1, box2):
    g_ymin, g_xmin, g_ymax, g_xmax = box1
    d_ymin, d_xmin, d_ymax, d_xmax = box2

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def confusion_vector_pass(detections_records,id_class_mapping):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1), dtype=np.int)
    for record in detections_records:
        detection_classes = record['detection']['classes']
        detection_boxes = record['detection']['bboxs']
        detection_scores = record['detection']['scores']
        detection_masks = record['detection']['masks'] 
        # detection_classes_TF = [x for x in detection_classes_TF if x == groundtruth_per_class[0]]
        # print("Final detection classes: {}".format(detection_classes))

        if len(detection_classes) == 0:
            confusion_matrix[confusion_matrix.shape[0] -1] += 1
        else:
            max_score = max(detection_scores)
            # print(detection_scores)
            detection_scores_TF = [1 if ((x == max_score) and (round(x, 2) >= 0.9)) else 0 for x in detection_scores]
            
            # print(detection_scores_TF)
            # detection_scores_TF =  [1 if round(x, 2) >= max_score else 0 for x in detection_scores]
            # print(detection_classes)

            detection_classes_TF = np.array(detection_classes) * np.array(detection_scores_TF) 
            # print(detection_classes_TF)

            detection_classes_TF = [x for x in detection_classes_TF if x != 0]
            detection_per_class = list(dict.fromkeys(detection_classes_TF))
            # print(detection_per_class)
            # print(detection_per_class)

            if len(detection_per_class) == 0:
                confusion_matrix[confusion_matrix.shape[0] - 1] += 1
            else:
                for j in range(len(detection_per_class)):
                    confusion_matrix[detection_per_class[j] - 1] += 1

    return confusion_matrix
def confusion_vector_classification(detections_records,id_class_mapping):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1), dtype=np.int)
    for record in detections_records:
        detection_classes = record['detection']['classes']
        detection_scores = record['detection']['scores']

        if len(detection_classes) == 0:
            confusion_matrix[confusion_matrix.shape[0] -1] += 1
        else:
    #     max_score = max(detection_scores)
    #     # print(detection_scores)
    #     detection_scores_TF = [1 if ((x == max_score) and (round(x, 2) >= 0.9)) else 0 for x in detection_scores]
        
    #     # print(detection_scores_TF)
    #     # detection_scores_TF =  [1 if round(x, 2) >= max_score else 0 for x in detection_scores]
    #     # print(detection_classes)

    #     detection_classes_TF = np.array(detection_classes) * np.array(detection_scores_TF) 
    #     # print(detection_classes_TF)

    #     detection_classes_TF = [x for x in detection_classes_TF if x != 0]
    #     detection_per_class = list(dict.fromkeys(detection_classes_TF))
    #     # print(detection_per_class)
    #     # print(detection_per_class)

    #     if len(detection_per_class) == 0:
    #         confusion_matrix[confusion_matrix.shape[0] - 1] += 1
    #     else:
            for j in range(len(detection_classes)):
                confusion_matrix[detection_classes[j] - 1] += 1
    return confusion_matrix

def confusion_matrix_classification(detections_records, id_class_mapping):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1, len(id_class_mapping) + 1), dtype=np.int)
    true_positives = []
    false_positives = []
    false_negatives = []

    for record in detections_records:
        groundtruth_classes = record['groundtruth']['classes']

        detection_classes = record['detection']['classes']
        detection_scores = record['detection']['scores']
        
        if len(detection_classes) == 0:
            confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
            detection_per_class = []
        else:
            detection_per_class = detection_classes

        for i in range(len(groundtruth_classes)):
            for j in range(len(detection_per_class)):
                confusion_matrix[groundtruth_classes[i] - 1][detection_per_class[j] - 1] += 1   

    return confusion_matrix

def confusion_matrix_image_base(detections_records, id_class_mapping, iou_threshold):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1, len(id_class_mapping) + 1), dtype=np.int)
    true_positives = []
    false_positives = []
    false_negatives = []

    for record in detections_records:
        groundtruth_boxes = record['groundtruth']['bboxs']
        groundtruth_classes = record['groundtruth']['classes']

        detection_classes = record['detection']['classes']
        detection_boxes = record['detection']['bboxs']
        detection_scores = record['detection']['scores']
        detection_masks = record['detection']['masks']
        
        # print("==== detection_class: {}".format(detection_classes))
        # print(len(detection_classes))
        matches = []
        # matches: (gt_index, pred_index, iou)
        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                if iou > iou_threshold:
                    matches.append([i, j, iou])
        matches = np.array(matches)

        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        if matches.shape[0] > 0:
            matches = np.array([matches[0]])
        # print(matches.shape[0])
        groundtruth_per_class = list(dict.fromkeys(groundtruth_classes))
        # detection_classes_TF = [x for x in detection_classes_TF if x == groundtruth_per_class[0]]
        # print("Final detection classes: {}".format(detection_classes))
        have_matches = matches.shape[0] > 0

        if len(detection_classes) == 0:
            confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
            detection_per_class = []
        else:
            max_score = max(detection_scores)

            detection_scores_TF =  [1 if x == max_score else 0 for x in detection_scores]

            detection_classes_TF = np.array(detection_classes) * np.array(detection_scores_TF) 

            detection_classes_TF = [x for x in detection_classes_TF if x != 0]
            
            detection_per_class = list(dict.fromkeys(detection_classes_TF))
        
        
        for i in range(len(groundtruth_per_class)):
            for j in range(len(detection_per_class)):
                # if have_matches:
                confusion_matrix[groundtruth_per_class[i] - 1][detection_per_class[j] - 1] += 1
                # else:
                    # confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
        # for i in range(len(groundtruth_boxes)):
        #     if have_matches:
        #         gt_i_matches = matches[matches[:, 0] == i]
        #         num_gt_i_matches = gt_i_matches.shape[0]

        #         if num_gt_i_matches == 1:
        #             match_pred_index = int(matches[matches[:, 0] == i, 1][0])

        #             if groundtruth_classes[i] == detection_classes[match_pred_index]:
        #                 pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
        #                                                                          detection_scores, None,
        #                                                                          match_pred_index, id_class_mapping)
        #                 true_positives.append({
        #                     'bbox': pred_box,
        #                     'score': pred_score,
        #                     'class': pred_class,
        #                 })

        #             else:
        #                 gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None,
        #                                                             i, id_class_mapping)

        #                 pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
        #                                                                          detection_scores, None,
        #                                                                          match_pred_index, id_class_mapping)
        #                 detection = {
        #                     'gt': {
        #                         'bbox': gt_box,
        #                         'class': gt_class,
        #                     },
        #                     'pred': {
        #                         'bbox': pred_box,
        #                         'class': pred_class,
        #                         'score': pred_score,
        #                     },
        #                 }
        #                 false_negatives.append(detection)

        #             # confusion_matrix[groundtruth_classes[i] - 1][detection_classes[match_pred_index] - 1] += 1

        #         else:
        #             # confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
        #             gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None,
        #                                                         i, id_class_mapping)

        #             detection = {
        #                 'gt': {
        #                     'bbox': gt_box,
        #                     'class': gt_class,
        #                 },
        #             }
        #             false_negatives.append(detection)

        #     else:
        #         # confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

        #         gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None, i,
        #                                                     id_class_mapping)
        #         detection = {
        #             'gt': {
        #                 'bbox': gt_box,
        #                 'class': gt_class,
        #             },
        #         }
        #         false_negatives.append(detection)

        # for i in range(len(detection_boxes)):
        #     if (have_matches and matches[matches[:, 1] == i].shape[0] == 0) or not have_matches:
        #         confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1

        #         pred_box, pred_class, pred_score, pred_mask = get_detection_info(detection_boxes, detection_classes,
        #                                                                          detection_scores, detection_masks, i,
        #                                                                          id_class_mapping)
        #         false_positives.append({
        #             'bbox': pred_box,
        #             'score': pred_score,
        #             'class': pred_class,
        #             'mask': pred_mask,
        #         })


    return confusion_matrix

def compute_confusion_matrix(detections_records, id_class_mapping, iou_threshold):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1, len(id_class_mapping) + 1), dtype=np.int)
    true_positives = []
    false_positives = []
    false_negatives = []
    for record in detections_records:
        groundtruth_boxes = record['groundtruth']['bboxs']
        groundtruth_classes = record['groundtruth']['classes']

        detection_classes = record['detection']['classes']
        detection_boxes = record['detection']['bboxs']
        detection_scores = record['detection']['scores']
        detection_masks = record['detection']['masks']

        
        matches = []
        # matches: (gt_index, pred_index, iou)
        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                if iou > iou_threshold:
                    matches.append([i, j, iou])
        matches = np.array(matches)
        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        have_matches = matches.shape[0] > 0
        for i in range(len(groundtruth_boxes)):
            if have_matches:
                gt_i_matches = matches[matches[:, 0] == i]
                num_gt_i_matches = gt_i_matches.shape[0]

                if num_gt_i_matches == 1:
                    match_pred_index = int(matches[matches[:, 0] == i, 1][0])

                    if groundtruth_classes[i] == detection_classes[match_pred_index]:
                        pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
                                                                                 detection_scores, None,
                                                                                 match_pred_index, id_class_mapping)
                        true_positives.append({
                            'bbox': pred_box,
                            'score': pred_score,
                            'class': pred_class,
                        })
                    else:
                        gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None,
                                                                    i, id_class_mapping)

                        pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
                                                                                 detection_scores, None,
                                                                                 match_pred_index, id_class_mapping)
                        detection = {
                            'gt': {
                                'bbox': gt_box,
                                'class': gt_class,
                            },
                            'pred': {
                                'bbox': pred_box,
                                'class': pred_class,
                                'score': pred_score,
                            },
                        }
                        false_negatives.append(detection)

                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[match_pred_index] - 1] += 1

                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1
                    gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None,
                                                                i, id_class_mapping)

                    detection = {
                        'gt': {
                            'bbox': gt_box,
                            'class': gt_class,
                        },
                    }
                    false_negatives.append(detection)

            else:
                confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

                gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None, i,
                                                            id_class_mapping)
                detection = {
                    'gt': {
                        'bbox': gt_box,
                        'class': gt_class,
                    },
                }
                false_negatives.append(detection)

        for i in range(len(detection_boxes)):
            if (have_matches and matches[matches[:, 1] == i].shape[0] == 0) or not have_matches:
                confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1

                pred_box, pred_class, pred_score, pred_mask = get_detection_info(detection_boxes, detection_classes,
                                                                                 detection_scores, detection_masks, i,
                                                                                 id_class_mapping)
                false_positives.append({
                    'bbox': pred_box,
                    'score': pred_score,
                    'class': pred_class,
                    'mask': pred_mask,
                })
    return confusion_matrix, true_positives, false_positives, false_negatives


def compute_confusion_matrix_on_one_record(record, id_class_mapping, iou_threshold):
    confusion_matrix = np.zeros(shape=(len(id_class_mapping) + 1, len(id_class_mapping) + 1), dtype=np.int)
    true_positives = []
    false_positives = []
    false_negatives = []

    groundtruth_boxes = record['groundtruth']['bboxs']
    groundtruth_classes = record['groundtruth']['classes']

    detection_classes = record['detection']['classes']
    detection_boxes = record['detection']['bboxs']
    detection_scores = record['detection']['scores']
    detection_masks = record['detection']['masks']

    matches = []
    # matches: (gt_index, pred_index, iou)
    for i in range(len(groundtruth_boxes)):
        for j in range(len(detection_boxes)):
            iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

            if iou > iou_threshold:
                matches.append([i, j, iou])

    matches = np.array(matches)
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending IOU. Removing duplicates doesn't preserve
        # our previous sort.
        matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    have_matches = matches.shape[0] > 0
    for i in range(len(groundtruth_boxes)):
        if have_matches:
            gt_i_matches = matches[matches[:, 0] == i]
            num_gt_i_matches = gt_i_matches.shape[0]

            if num_gt_i_matches == 1:
                match_pred_index = int(matches[matches[:, 0] == i, 1][0])

                if groundtruth_classes[i] == detection_classes[match_pred_index]:
                    pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
                                                                             detection_scores, None,
                                                                             match_pred_index, id_class_mapping)
                    true_positives.append((pred_box, pred_score, pred_class))
                    true_positives.append({
                        'bbox': pred_box,
                        'score': pred_score,
                        'class': pred_class,
                    })
                else:
                    gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None,
                                                                i, id_class_mapping)

                    pred_box, pred_class, pred_score, _ = get_detection_info(detection_boxes, detection_classes,
                                                                             detection_scores, None,
                                                                             match_pred_index, id_class_mapping)
                    detection = {
                        'gt': {
                            'bbox': gt_box,
                            'class': gt_class,
                        },
                        'pred': {
                            'bbox': pred_box,
                            'class': pred_class,
                            'score': pred_score,
                        },
                    }
                    false_negatives.append(detection)

                confusion_matrix[groundtruth_classes[i] - 1][detection_classes[match_pred_index] - 1] += 1

        else:
            confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

            gt_box, gt_class, _, _ = get_detection_info(groundtruth_boxes, groundtruth_classes, None, None, i,
                                                        id_class_mapping)
            detection = {
                'gt': {
                    'bbox': gt_box,
                    'class': gt_class,
                },
            }
            false_negatives.append(detection)

    for i in range(len(detection_boxes)):
        if (have_matches and matches[matches[:, 1] == i].shape[0] == 0) or not have_matches:
            confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1

            pred_box, pred_class, pred_score, pred_mask = get_detection_info(detection_boxes, detection_classes,
                                                                             detection_scores, detection_masks, i,
                                                                             id_class_mapping)
            false_positives.append({
                'bbox': pred_box,
                'score': pred_score,
                'class': pred_class,
                'mask': pred_mask,
            })

    return confusion_matrix, true_positives, false_positives, false_negatives


def compute_escape_overkill(detection_record):
    tp = fn = tn = fp = 0
    for record in detection_record:
        gt_record = record['groundtruth']
        pred_record = record['detection']

        lens_gt_classes = set(gt_record['classes'])
        lens_pred_classes = set(pred_record['classes'])

        # print('gt: %s' % lens_gt_classes)
        # print('pred: %s' % lens_pred_classes)
        fn_classes = Counter()
        if lens_gt_classes:
            if lens_gt_classes.issubset(lens_pred_classes):
                tp += 1
            else:
                fn += 1
                fn_classes = Counter(lens_gt_classes) - Counter(lens_pred_classes)

        else:
            if lens_pred_classes:
                fp += 1
            else:
                tn += 1
    return {
        'tp': tp,
        'fn': fn,
        'tn': tn,
        'fp': fp,
        'fn_classes': fn_classes,
    }


def get_detection_info(detection_boxes, detection_classes, detection_scores, detection_masks, i, id_class_mapping):
    detection_box = detection_boxes[i]
    if detection_scores is None:
        detection_score = None
    else:
        detection_score = detection_scores[i]
    if detection_masks is None:
        detection_mask = None
    else:
        detection_mask = detection_masks[i]
    detection_class = id_class_mapping[detection_classes[i]]

    return detection_box, detection_class, detection_score, detection_mask
def get_record_from_pass_result(detect_result):
    record= {
        'detection':{
        'bboxs':[],
        'classes':[],
        'scores':[],
        'masks':[]
        }
    }
    # Get only result
    rois_lists=detect_result['rois'].tolist()
    for rois_list in rois_lists:
        record['detection']['bboxs'].append((rois_list[0],rois_list[1],rois_list[2],rois_list[3]))
    # print(record['detection']['bboxs'])
    class_id_lists=detect_result['class_ids'].tolist()
    for class_id_list in class_id_lists:
        record['detection']['classes'].append(class_id_list)
    score_lists=detect_result['scores'].tolist()
    for score_list in score_lists:
        record['detection']['scores'].append(score_list)
        record['detection']['masks'].append(None)
    # print(record)
    return record

def get_record_from_txt_and_result(labels ,detect_class, detect_score):
    record={
    'groundtruth':{
        'classes':[]
    },
    'detection':{
        'classes':[],
        'scores':[]
    }
}
    record['groundtruth']['classes'].append(labels)
    record['detection']['classes'].append(detect_class)
    record['detection']['scores'].append(detect_score)
    return record

def get_record_from_result_only(detect_class, detect_score):
    record={
    'detection':{
        'classes':[],
        'scores':[]
        }
    }
    record['detection']['classes'].append(detect_class)
    record['detection']['scores'].append(detect_score)
    return record

def get_record_from_json_and_result(jsonconfig_path,detect_result, config_json):
    with open(config_json) as f:
        obj = json.load(f)
    class_name = obj['ClassName']
    # print("Hello")
    record={
    'groundtruth':{
        'bboxs':[],
        'classes':[]
    },
    'detection':{
        'bboxs':[],
        'classes':[],
        'scores':[],
        'masks':[]
    }
}
    param = loadJSON_label.LoadJSON_label(jsonconfig_path)
    for key,value in param.regions.items():
        record['groundtruth']['bboxs'].append((min(value['List_Y']),min(value['List_X']),max(value['List_Y']),max(value['List_X'])))
        for index, className in enumerate(class_name):
            if(param.classId[int(key)]== className):
                id_class=index +1
        record['groundtruth']['classes'].append(id_class)
    # print(record)


    rois_lists=detect_result['rois'].tolist()
    for rois_list in rois_lists:
        record['detection']['bboxs'].append((rois_list[0],rois_list[1],rois_list[2],rois_list[3]))
    # print(record['detection']['bboxs'])
    class_id_lists=detect_result['class_ids'].tolist()
    # print(class_id_lists)
    for class_id_list in class_id_lists:
        record['detection']['classes'].append(class_id_list)
    score_lists=detect_result['scores'].tolist()
    for score_list in score_lists:
        record['detection']['scores'].append(score_list)
        record['detection']['masks'].append(None)
    # print(record)
    return record

def evaluate_img_classification(record, config_txt, mode):
    detection_records = record
    # with open('%s/classes.txt' % config_txt) as f:
    #     classes = f.readlines()
    with open('%s/config_KLA.json' % config_txt) as f:
        data = json.load(f)
    classes = data["ClassName"]
    id_class_mapping ={ i + 1 : classes[i] for i in range(0, len(classes) ) }
    if mode == 1:
        confusion_matrix = confusion_matrix_classification(detection_records,
                                            id_class_mapping=id_class_mapping)
        # logger.info('True positives: %s' % true_positives)
        # logger.info('False positives: %s' % false_positives)
        # logger.info('False negatives: %s' % false_negatives)
        logger.info("Confusion Matrix:")
        logger.info('\n%s' % confusion_matrix.astype(int))

        # Record-based (Image-based) evaluation
        lens_evaluation = compute_escape_overkill(record)
        logger.info(lens_evaluation)
        return lens_evaluation
    else:
        confusion_matrix = confusion_vector_classification(detection_records,
                                            id_class_mapping=id_class_mapping,)
        logger.info('\n%s' % confusion_matrix.astype(int))


def evaluate_img(record, mode, config_json):
    detection_records = record
    with open(config_json) as f:
        obj = json.load(f)   

    id_class_mapping ={ i + 1 : obj['ClassName'][i] for i in range(0, len(obj['ClassName']) ) }
    # print(id_class_mapping)
    # id_class_mapping = {
    #     1: 'PATT_FLAKE',
    #     2: 'CU_PUDDLE', 
    #     3: 'BLK_DEP_DEF',
    # }
    # Defect-based evaluation
    # confusion_matrix, true_positives, false_positives, false_negatives = compute_confusion_matrix(detection_records,
    #                                                                                               id_class_mapping=id_class_mapping,
    #                                                                                               iou_threshold=0.5)
    # Image-based evaluation
    if mode == 0 :
        confusion_matrix = confusion_matrix_image_base(detection_records,
                                        id_class_mapping=id_class_mapping,
                                        iou_threshold=0.5)
        # logger.info('True positives: %s' % true_positives)
        # logger.info('False positives: %s' % false_positives)
        # logger.info('False negatives: %s' % false_negatives)
        logger.info("Confusion Matrix:")
        logger.info('\n%s' % confusion_matrix.astype(int))

        # Record-based (Image-based) evaluation
        lens_evaluation = compute_escape_overkill(record)
        logger.info(lens_evaluation)
        return lens_evaluation
    else:
        confusion_matrix = confusion_vector_pass(detection_records,
                                            id_class_mapping=id_class_mapping,)
        logger.info('\n%s' % confusion_matrix.astype(int))

# if __name__ == '__main__':
#     record = {
#         'groundtruth': {
#             'bboxs': [(0, 0, 2, 2), (2, 2, 3, 3), (6, 6, 7, 7)],  # bbox = (min(ys), min(xs), max(ys), max(xs))
#             'classes': [1, 3, 2],
#         },
#         'detection': {
#             'bboxs': [(0, 0, 2, 2), (2, 2, 4, 4), (4, 4, 5, 5)],  # bbox: ['rois'][i].tolist()
#             'classes': [1, 3, 2],
#             'scores': [0.9, 0.8, 0.9],
#             'masks': [None, None, None],
#         },
#     }
#     detection_records = [record]
#     id_class_mapping = {
#         1: 'Class A',
#         2: 'Class B',
#         3: 'Class C',
#     }

#     # Defect-based evaluation
#     confusion_matrix, true_positives, false_positives, false_negatives = compute_confusion_matrix(detection_records,
#                                                                                                   id_class_mapping=id_class_mapping,
#                                                                                                   iou_threshold=0.1)
#     logger.info('True positives: %s' % true_positives)
#     logger.info('False positives: %s' % false_positives)
#     logger.info('False negatives: %s' % false_negatives)
#     logger.info("Confusion Matrix:")
#     logger.info('\n%s' % confusion_matrix.astype(int))

#     # Record-based (Image-based) evaluation
#     lens_evaluation = compute_escape_overkill(record)

#     logger.info(lens_evaluation)
