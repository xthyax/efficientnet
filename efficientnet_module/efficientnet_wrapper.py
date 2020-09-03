import os
os.environ['TF_KERAS']='1'
import json
import time
from pathlib import Path

# from keras_applications.imagenet_utils import _obtain_input_shape
import cv2
import numpy as np
import pandas as pd
# import keras
# from efficientnet.keras import *
# from tensorflow.keras.models import load_model
# from keras.engine.saving import model_from_json
# from keras.utils import multi_gpu_model
# from keras import backend as K
# from keras import optimizers

from efficientnet.tfkeras import *
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import multi_gpu_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import optimizers

from keras_lookahead import Lookahead
from keras_radam import RAdam

from .callbacks import SaveMultiGPUModelCheckpoint, CustomCallback
from .data_generator import DataGenerator
# from .utils import cou
from .utils import multi_threshold, recursive_glob, compute_class_weight, recursive_folder, config_dump, load_and_crop, \
    get_dataframe, get_z_score_info, get_z_score, SplitDataFrameToTrainAndTest, get_dataframe_one
from .focal_loss import focal_loss
# from .parallel_model import ParallelModel
# from efficientnet_module.modules.config import Optimizer_, FREEZE
# from efficientnet_module.modules import config

import glob
import tqdm
import itertools
import shutil
import xlsxwriter

# import xgboost as xgb
# from xgboost import XGBClassifier
from sklearn.ensemble import *
from sklearn.neighbors import *
from sklearn.tree import *
from tensorboardX import SummaryWriter
from keras.callbacks import ModelCheckpoint

class EfficientNetWrapper:
    def __init__(self, config):
        self.classes = config.CLASS_NAME
        self.input_size = config.INPUT_SIZE
        self.binary_option = config.BINARY
        self.failClasses = config.FAIL_CLASSNAME
        self.passClasses = config.PASS_CLASSNAME
        self.keras_model = None
        self.X = None
        self.Y = None
        self.num_of_classes = len(self.classes)
        self.data = None
        self.train_generator = None
        self.val_generator = None
        self.test_generator = None
        self.evaluate_generator = None
        self.id_class_mapping = None
        self.class_weights = None
        self.graph = tf.get_default_graph()
        self.session = K.get_session()
        self.config = config
        self.ensemble_model = None

    def _build_model(self):
        # initializer = keras.initializers.glorot_uniform(seed=1)
        try:
            model_class = {
                'B0': EfficientNetB0,
                'B1': EfficientNetB1,
                'B2': EfficientNetB2,
                'B3': EfficientNetB3,
                'B4': EfficientNetB4,
                'B5': EfficientNetB5,
                'B6': EfficientNetB6,
                'B7': EfficientNetB7,
                'L2': EfficientNetL2
            }[self.config.ARCHITECTURE]
        except KeyError:
            raise ValueError('Invalid Classification Settings')

        # Redundant RGB channels for grayscale input
        base_model = model_class(\
            input_shape=(self.input_size, self.input_size, 3),\
            # input_shape=( 380, 380, 3),\
            weights= self.config.WEIGHT_PATH if self.config.WEIGHT_PATH is not None else 'imagenet',\
            include_top=False)

        freeze = None
        if freeze is None:
            pass
        else:
            for layer in base_model.layers:
                layer.trainable = not freeze
        
        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(self.num_of_classes, activation='softmax'\
            # ,kernel_initializer=initializer\
            )(x)
        
        return keras.models.Model(inputs=[base_model.input], outputs=[output])

    def prepare_data(self):
        # TODO: Merge with Front-end and dev

        self.load_classes()

        list_Directory = [
            os.path.join(self.config.DATASET_PATH, 'Train'),
            os.path.join(self.config.DATASET_PATH, 'Validation'),
            os.path.join(self.config.DATASET_PATH, 'Test'),
        ]

        list_Generator = []
        for diRectory in list_Directory.copy():
            if not os.path.exists(diRectory) or len(os.listdir(diRectory)) == 0:
                list_Directory.remove(diRectory)

        for diRectory in list_Directory:
            generator = DataGenerator(diRectory, self.config.BATCH_SIZE * self.config.GPU_COUNT,\
                self.classes, self.failClasses, self.passClasses,\
                self.input_size, self.binary_option, augmentation=self.config.AU_LIST if "train" in diRectory.lower() else None )
        
            list_Generator.append(generator)
        check_train = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "train" in set_path.lower()]]]
        self.train_generator = check_train[0] if len(check_train) > 0 else None
        
        check_val = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "validation" in set_path.lower()]]]
        self.val_generator = check_val[0] if len(check_val) > 0 else None
        
        check_test = [list_Generator[s_value] for s_value in [value for value in [list_Directory.index(set_path) for set_path in list_Directory if "test" in set_path.lower()]]]
        self.test_generator = check_test[0] if len(check_test) > 0 else None

        self.evaluate_generator = DataGenerator(list_Directory, self.config.BATCH_SIZE,\
            self.classes, self.failClasses, self.passClasses, \
            self.input_size, self.binary_option)

        self.class_weights = compute_class_weight(self.train_generator.metadata.values())

    def lossFunc_chosen(self):
        if len(self.classes) > 2 :
            return keras.losses.categorical_crossentropy
        else:
            return keras.losses.binary_crossentropy

        # try:
        #     loss_dict = {
        #         'binary': 'binary_crossentropy',
        #         'category' : 'categorical_crossentropy',
        #         'focal' : EfficientNetWrapper.loss_func(self.class_weights)
        #     }[option]
        #     return loss_dict
        # except KeyError:
        #     raise ValueError("Invalid loss function")

    def load_classes(self):
        if self.binary_option:
            init_class = ['Reject','Pass']
            self.classes = init_class
            self.num_of_classes = len(init_class)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(init_class)}

        else:

            self.num_of_classes = len(self.classes)
            self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(self.classes)}

    def load_classes_kaggle(self, classes):
        self.classes = classes
        self.num_of_classes = len(self.classes)
        self.id_class_mapping = {class_id: class_name for class_id, class_name in enumerate(self.classes)}

    def optimizer_chosen(self):
        # TODO: Tune optimizers params
        try:
            optimizer_dict = {
                'sgd': optimizers.SGD(lr=self.config.LEARNING_RATE, momentum=self.config.LEARNING_MOMENTUM),
                'adam': optimizers.Adam(lr=self.config.LEARNING_RATE),
                'rmsprop': optimizers.RMSprop(lr=self.config.LEARNING_RATE,decay=self.config.WEIGHT_DECAY),
                'nadam': optimizers.Nadam(lr=self.config.LEARNING_RATE),
                'radam': RAdam(),
                'ranger': Lookahead(RAdam()),
            }[self.config.OPTIMIZER.lower()]

            return optimizer_dict
        except KeyError:
            raise ValueError("Invalid optimizer")
        
    def train(self):
        train_checkpoint_dir = self.config.LOGS_PATH
        os.makedirs(train_checkpoint_dir,exist_ok=True)
        # print(self.config)
        # config_dump(train_checkpoint_dir, self.config)

        self.keras_model = self._build_model()

        # train
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=train_checkpoint_dir)

        if self.config.GPU_COUNT > 1:
            model = multi_gpu_model(self.keras_model, gpus=self.config.GPU_COUNT)
        elif self.config.GPU_COUNT == 1:
            model = self.keras_model
        else:
            raise ValueError("Invalid 'gpu_count' value")

        optimizer = self.optimizer_chosen()

        # lr_callback = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: log.update{'lr': K.eval()})

        # model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # my_callback = ModelCheckpoint(train_checkpoint_dir + '/ep{epoch:04d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=False)
        checkpoint_callback = SaveMultiGPUModelCheckpoint(self.keras_model, train_checkpoint_dir)

        model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])
        model.fit_generator(self.train_generator, epochs=self.config.EPOCH, validation_data=self.val_generator, max_queue_size=10,
                            workers=1, callbacks=[checkpoint_callback, tensorboard_callback])

    def load_weight(self):
        self.keras_model = load_model(self.config.WEIGHT_PATH, compile=False)
        self.load_classes()

    def predict_one(self, img, return_all):
        # resized_img = cv2.resize(img, (self.input_size, self.input_size))
        # resized_img = image_read(img, self.input_size)
        input_data = np.array([img])
        X = preprocess_input(input_data)
        # Using ensemble model ==============================
        # X_data = self.get_feature_one(img)
        # ensemble_prop = self.ensemble_prediction(X_data)
        # ===================================================
        # print(f"[DEBUG] ensemble propability : {ensemble_prop[0]}")
        with self.graph.as_default():
            with self.session.as_default():
                Y = self.keras_model.predict(X)
        # Combine with ensemble model========================
        # Y = Y * 0.7 + ensemble_prop * 0.3
        # ===================================================
        if self.config.CLASS_THRESHOLD is None or len(self.config.CLASS_THRESHOLD) == 0:
            Y_class_id = np.argmax(Y, axis=-1)
            Y_score = np.max(Y, axis=-1)
        else:
            ret = multi_threshold(Y, self.config.CLASS_THRESHOLD)
            if ret is None:
                classID = -1
                className = "Unknown"
                scores = Y[0]
                return classID, scores, className
            else:
                Y_class_id, Y_score = ret

        Y_class_name = self.id_class_mapping[Y_class_id[0]]
        # print(f"[DEBUG] Y_class_id: {Y_class_id}")
        # print(f"[DEBUG] Y_score: {Y_score}")
        # print(f"[DEBUG] all scores: {Y[0]}")
        if return_all :
            return Y[0]
        else:
            return Y_class_id[0], Y_score[0], Y_class_name

    def evaluate(self, subset='test'):
        # TODO: make evaluate for all set instead of just test set
        optimizer = self.optimizer_chosen()
        self.keras_model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])
        # self.keras_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        subset_generator = {'train': self.train_generator, 'val': self.val_generator, 'test': self.test_generator}[
            subset]
        result = self.keras_model.evaluate_generator(subset_generator, verbose=1)
        print('Loss: %s' % result[0])
        print('Accuracy: %s' % result[1])

    def resume_training(self):
        epoch = 0
        strategy =  tf.distribute.Strategy()
        print("[INFO] Number of devices: {}".format(strategy.num_replicas_in_sync))
        optimizer = self.optimizer_chosen()
        strategy = tf.distribute.MirroredStrategy()
        train_checkpoint_dir = self.config.LOGS_PATH

        if "startingmodel.h5" in self.config.WEIGHT_PATH:
            if self.config.GPU_COUNT > 1:
                with strategy.scope():
                    self.keras_model = self._build_model()
                    model = self.keras_model
                    model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])
                # model = multi_gpu_model(self.keras_model, gpus=self.config.GPU_COUNT)
            elif self.config.GPU_COUNT == 1:
                self.keras_model = self._build_model()
                model = self.keras_model
                model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])
            else:
                raise ValueError("Invalid 'gpu_count' value")
        else:
            if self.config.GPU_COUNT > 1:
                with strategy.scope():
                    self.load_weight()
                    model = self.keras_model
                    model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])

            elif self.config.GPU_COUNT == 1:
                self.load_weight()
                model = self.keras_model
                model.compile(optimizer=optimizer, loss=self.lossFunc_chosen(), metrics=['accuracy'])
            else:
                raise ValueError("Invalid 'gpu_count' value")
        # train
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=train_checkpoint_dir)
        
        writer = SummaryWriter(log_dir=train_checkpoint_dir)
        custom_callback = CustomCallback(writer ,\
            self.evaluate_generator, self.classes,\
            ['Reject'] if self.binary_option else self.failClasses, \
            ['Pass'] if self.binary_option else self.passClasses)

        # custom_callback = CustomCallback(tensorboard_callback,\
        #     self.evaluate_generator, self.classes, \
        #     ['Reject'] if self.binary_option else self.failClasses, \
        #     ['Pass'] if self.binary_option else self.passClasses)

        checkpoint_callback = SaveMultiGPUModelCheckpoint(self.keras_model, train_checkpoint_dir)

        model.fit(x=self.train_generator, epochs=self.config.NO_EPOCH, validation_data=self.val_generator,\
                max_queue_size=10, workers=1, callbacks=[checkpoint_callback, tensorboard_callback, custom_callback],\
                initial_epoch=epoch)

        model.fit_generator(self.train_generator, epochs=self.config.NO_EPOCH, validation_data=self.val_generator,
                            max_queue_size=10, workers=1, callbacks=[checkpoint_callback, tensorboard_callback, custom_callback],
                            initial_epoch=epoch)

    def labelling_raw_data(self):
        path = [
            os.path.join(self.config.DATASET_PATH)
        ]
        result_path = [
            os.path.join("_Labelled","Reject"),
            os.path.join("_Labelled","Pass"),
            os.path.join("_Labelled","Unclear")
        ]
        for sub_path in path:
            print(f"[DEBUG] Labelling: {sub_path}")
            images_list = []
            for image in glob.glob(sub_path + "/*.bmp"):
                images_list.append(image)

        with tqdm.tqdm(total=len(images_list)) as pbar:
            for image_index, image_path in itertools.islice(enumerate(images_list), len(images_list)):

                img, gt_name = load_and_crop(image_path, self.config.INPUT_SIZE)
                pred_id, pred_score, pred_name = self.predict_one(img, 0)
                all_scores = self.predict_one(img, 1)
                image_name = image_path.split("\\")[-1]
                print(f"[DEBUG] image id:\t{image_name}")
                print(f"[DEBUG] all scores:\t{all_scores}")
                print(f"[DEBUG] pred_id: {pred_id} - pred_score: {pred_score} -  pred_name: {pred_name}")

                if self.binary_option:
                    
                    if (pred_id == 1 or pred_id == -1) and pred_score >= 0.8:   # Pass
                        Pass_path = os.path.join("_Labelled","Pass")
                        os.makedirs(Pass_path, exist_ok=True)

                        shutil.copy(image_path, os.path.join(Pass_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Pass"]

                        # with open(os.path.join(Pass_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)

                    elif pred_id == 0 and pred_score >= 0.8:                    # Reject
                        Reject_path = os.path.join("_Labelled","Reject")
                        os.makedirs(Reject_path, exist_ok=True)
                        
                        shutil.copy(image_path, os.path.join(Reject_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Burr"]

                        # with open(os.path.join(Reject_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)
                    else:                                                       # Unknow
                        Unclear_path = os.path.join("_Labelled","Unknow")
                        os.makedirs(Unclear_path, exist_ok=True)
                        
                        shutil.copy(image_path, os.path.join(Unclear_path,image_name))
                        json_path = image_path + ".json"

                        # with open(json_path, encoding='utf-8') as json_file:
                        #     json_data = json.load(json_file)
                        # json_data["classId"] = ["Unclear"]

                        # with open(os.path.join(Unclear_path,image_name) + ".json", "w") as bmp_json:
                        #     json.dump(json_data, bmp_json)
                else:
                    gt_id = self.classes.index(gt_name)

                pbar.update()

        logger.info("Done")

    def confusion_matrix_evaluate(self):
        path =  [
            # os.path.join(self.config.DATASET_PATH,"3party_part1_disagree_reviewed"),    #Hardcode
            # os.path.join(self.config.DATASET_PATH,"agree_part2"),                       #Hardcode
            # os.path.join(self.config.DATASET_PATH,"agree_part3"),                       #Hardcode
            # os.path.join(self.config.DATASET_PATH,"disagree_defect_type_part2"),        #Hardcode
            # os.path.join(self.config.DATASET_PATH,"disagree_defect_type_part3"),        #Hardcode
            os.path.join(self.config.DATASET_PATH,"Train\\OriginImage"),    #Hardcode
            os.path.join(self.config.DATASET_PATH,"Validation"),            #Hardcode
            os.path.join(self.config.DATASET_PATH,"Test")                   #Hardcode
        ]
        # result_path = [
        #     os.path.join("_Result","UK"),
        #     os.path.join("_Result","OK")
        # ]
        # for sub_result in result_path:
        #     os.makedirs(sub_result, exist_ok=True)
        workbook = xlsxwriter.Workbook("_model_result.xlsx")

        cell_format = workbook.add_format()
        cell_format.set_align('center')
        cell_format.set_align('vcenter')

        highlight_format = workbook.add_format()
        highlight_format.set_align('center')
        highlight_format.set_align('vcenter')
        highlight_format.set_bg_color("red")
        # highlight_format.set_font_color

        Header = ["image_id","Image","Label","Predict"]
        Header.extend(self.classes)
        Header.append("Underkill")
        Header.append("Overkill")
        # Init ensemble model
        # self.get_ensemble_model()
        fail_class_index = [self.classes.index(class_) for class_ in self.failClasses]
        pass_class_index = [self.classes.index(class_) for class_ in self.passClasses]
        for sub_path in path:
            print(f"[DEBUG] Evaluating: {sub_path}")
            image_list = []
            for image in glob.glob(sub_path + "/*.bmp"):
                image_list.append(image)
            # print(image_list)
            # break

            start_row = 0
            start_column = 1
            worksheet = workbook.add_worksheet(sub_path.split("\\")[-1])
            worksheet.write_row( start_row, start_column, Header, cell_format)
            worksheet.set_column("C:C",10)
            confusion_matrix = np.zeros(shape=(len(self.id_class_mapping) + 1, len(self.id_class_mapping) + 1), dtype=np.int)
            
            with tqdm.tqdm(total=len(image_list)) as pbar:
                for image_index, image_path in itertools.islice(enumerate(image_list), len(image_list)):
                    Data = [0] * len(Header)
                    start_row += 1
                    worksheet.set_row(start_row, 60)
                    underkill_overkill_flag = 0
                    img, gt_name = load_and_crop(image_path, self.input_size)
                    pred_id, pred_score, pred_name = self.predict_one(img, 0)
                    all_scores = self.predict_one(img, 1)
                    if self.binary_option:

                        gt_name = 'Reject' if gt_name in self.failClasses else 'Pass'
                        gt_id = self.classes.index(gt_name)
                        image_name = image_path.split("\\")[-1]
                        if gt_id == 0 and (pred_id == 1 or pred_id == -1):  # Underkill
                            underkill_path = os.path.join("_Result",image_path.split("\\")[-2],"UK")
                            os.makedirs(underkill_path, exist_ok=True)
                            image_output_path = os.path.join(underkill_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(underkill_path,image_name+".json"))
                            underkill_overkill_flag = -1
                        elif gt_id == 1 and pred_id == 0:                   # Overkill
                            overkill_path = os.path.join("_Result",image_path.split("\\")[-2],"OK")
                            os.makedirs(overkill_path, exist_ok=True)
                            image_output_path = os.path.join(overkill_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(overkill_path,image_name+".json"))
                            underkill_overkill_flag = 1
                        else:                                               # Correct result
                            result_path = os.path.join("_Result", image_path.split("\\")[-2])
                            os.makedirs(result_path, exist_ok=True)
                            image_output_path = os.path.join(result_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(result_path,image_name + ".json"))
                    else:

                        gt_id = self.classes.index(gt_name)
                        image_name = image_path.split("\\")[-1]
                        if gt_id in fail_class_index and pred_id in pass_class_index:   # Underkill
                            underkill_path = os.path.join("_Result",image_path.split("\\")[-2],"UK")
                            os.makedirs(underkill_path, exist_ok=True)
                            image_output_path = os.path.join(underkill_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(underkill_path,image_name+".json"))
                            underkill_overkill_flag = -1
                        elif gt_id in pass_class_index and pred_id in fail_class_index:  # Overkill
                            overkill_path = os.path.join("_Result",image_path.split("\\")[-2],"OK")
                            os.makedirs(overkill_path, exist_ok=True)
                            image_output_path = os.path.join(overkill_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(overkill_path,image_name+".json"))
                            underkill_overkill_flag = 1
                        else:                                               # Correct result
                            result_path = os.path.join("_Result", image_path.split("\\")[-2])
                            os.makedirs(result_path, exist_ok=True)
                            image_output_path = os.path.join(result_path,image_name)
                            cv2.imwrite(image_output_path, img)
                            shutil.copy(image_path + ".json", os.path.join(result_path,image_name + ".json"))
                    
                    confusion_matrix[gt_id][pred_id] += 1
                    
                    Data[0] = image_name.split(".")[0]
                    Data[2] = gt_name
                    Data[3] = pred_name
                    Data[4:4+len(self.classes)] = all_scores
                    Data[-2] = True if underkill_overkill_flag == -1 else False
                    Data[-1] = True if underkill_overkill_flag == 1 else False

                    for index, info in enumerate(Data):
                        
                        excel_format = highlight_format if (Data[index] == True and isinstance(Data[index],bool)) else cell_format

                        worksheet.insert_image(start_row, index + 1, image_output_path, {'x_scale': 0.5,'y_scale': 0.5, 'x_offset': 5, 'y_offset': 5,'object_position':1}\
                            ) if index == 1 else worksheet.write(start_row, index + 1, Data[index], excel_format)
                    pbar.update()
            header = [{'header': head} for head in Header]

            worksheet.add_table(0, 1, start_row, len(Header), {'columns':header})
            worksheet.freeze_panes(1,0)
            worksheet.hide_gridlines(2)
            

            print(f"Confusion matrix of {sub_path}")
            print('\n%s' % confusion_matrix.astype(int))

        workbook.close()
    def checking_models(self):
        model_path_ls = []
        
        for path_trained_model in glob.glob(os.path.join(self.config.WEIGHT_PATH, "*.h5")):
            model_path_ls.append(path_trained_model)

        fail_ls_class =  ['Reject'] if self.binary_option else self.failClasses
        pass_ls_class = ['Pass'] if self.binary_option else self.passClasses

        model_info = {
                "path": [],
                "model_name": [],
                "Underkill_rate": [],
                "Overkill_rate": []
            }
        min_OK_rate = 100
        for trained_model in model_path_ls:
            self.config.WEIGHT_PATH = trained_model
        
            self.load_weight()
            fail_class_index = [self.classes.index(item_class) for item_class in fail_ls_class]
            pass_class_index = [self.classes.index(item_class) for item_class in pass_ls_class]
            
            param = {
                "FN": 0,
                "TP": 0,
                "FP": 0,
                "TN": 0
            }
            
            print("====================================")
            print(f"Testing model : {self.config.WEIGHT_PATH}.....")
            
            with self.graph.as_default():
                with self.session.as_default():
                    for _, sample in enumerate(self.evaluate_generator):
                        x_data, y_data = sample
                        
                        x_result = self.keras_model.predict(x_data)

                        predict_classes = x_result.argmax(axis=-1)

                        for i in range(len(x_data)):
                            # Check groundtruth
                            y_current = y_data[i].tolist()

                            if y_current.index(1) in fail_class_index:
                                if predict_classes[i] == y_current.index(1)\
                                    or predict_classes[i] in fail_class_index:
                                    param['TP'] += 1
                                else:
                                    param['FN'] += 1
                            else:
                                if predict_classes[i] == y_current.index(1)\
                                    or predict_classes[i] in pass_class_index:
                                    param['TN'] += 1
                                else:
                                    param['FP'] += 1

                    UK_rate = (param['FN'] / (param['TP'] + param['FN'])) * 100
                    OK_rate = (param['FP'] / (param['TN'] + param['FP'])) * 100

                    print(f"Underkill rate: {UK_rate} %")
                    print(f"Overkill rate: {OK_rate} %")
                    print("====================================")
                    print("\n")
                    if UK_rate == 0.0 and OK_rate <= min_OK_rate:
                        # os.path.join(*(x.split(os.path.sep)[2:]))
                        model_info["path"] = os.path.join(*(self.config.WEIGHT_PATH.split("\\")[:-1]))
                        model_info["model_name"] = [self.config.WEIGHT_PATH.split("\\")[-1]]
                        model_info['Underkill_rate'] = [UK_rate]
                        model_info['Overkill_rate'] = [OK_rate]
                        min_OK_rate = OK_rate

        with open("model_info.json", "w") as model_json:
            json.dump(model_info, model_json)

    def get_handcraft_feature(self):
        Data_Path = os.path.join(self.config.DATASET_PATH, "Train\\OriginImage")
        get_dataframe([Data_Path], self.failClasses)
      
    def get_ensemble_model(self):
        TrainDF_Model = pd.read_pickle("TrainDF_Model.pkl")
        # if self.binary_option:
        #     TrainDF_Model['Class'] = ['Reject' if current_class in self.failClasses else current_class for current_class in TrainDF_Model['Class'].tolist()]
        get_z_score_info(TrainDF_Model)
        Mean_std_DF = pd.read_pickle("Mean_std_value.pkl")
        get_z_score(Mean_std_DF, [TrainDF_Model])

        train_data = TrainDF_Model.copy()
        train_data = train_data.drop("Path", axis=1)
        data_train, target_train, _, _ = SplitDataFrameToTrainAndTest(train_data, 1, 'Class')
        X_train = data_train
        y_train = target_train
        y_train = [self.classes.index(target) for target in y_train['Class'].tolist()]
        X_train = X_train.apply(pd.to_numeric, errors = 'coerce')
        Classifier_ls = [\
        BaggingClassifier(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', min_samples_split=12), n_estimators=1000, max_samples=0.8, max_features=1.0, random_state=0), \
        RandomForestClassifier(criterion='entropy', n_estimators=1000, random_state=0, class_weight='balanced', min_samples_split=12, max_samples=0.8), \
        AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy', class_weight='balanced', min_samples_split=12), n_estimators=2000, learning_rate=1.0, random_state=0), \
        XGBClassifier( n_estimators=200, random_state=0, learning_rate=0.3, tree_method='gpu_hist', gpu_id=0),\
        ]
        self.ensemble_model = VotingClassifier(estimators=[('bg', Classifier_ls[0]), ('rf',Classifier_ls[1]),('ada',Classifier_ls[2]),('xgb',Classifier_ls[3])], voting='soft', weights=[1,1,1,1])
        # print(TrainDF_Model.head())
        self.ensemble_model.fit(X_train, y_train)

    def get_feature_one(self, img):
        feature_dataframe = get_dataframe_one(img)
        Mean_std_DF = pd.read_pickle("Mean_std_value.pkl")
        get_z_score(Mean_std_DF, [feature_dataframe])
        features_data = feature_dataframe.copy()
        X_data = features_data
        X_data = X_data.apply(pd.to_numeric, errors = 'coerce')
        # print(f"X_data:\n{X_data}")
        return X_data

    def ensemble_prediction(self, input_feature):
        pred_score = self.ensemble_model.predict_proba(input_feature)
        return pred_score
    @staticmethod
    def loss_func(class_weight):
        # TODO: change loss function to be more dynamic
        def loss(y_true, y_pred):
            return focal_loss(y_true, y_pred, class_weight)
        
        return loss