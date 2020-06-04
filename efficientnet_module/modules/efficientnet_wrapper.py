import os
import json
import time
from pathlib import Path

# from keras_applications.imagenet_utils import _obtain_input_shape
import cv2
import keras
import numpy as np
from efficientnet.keras import *
import tensorflow as tf
from keras.models import load_model
from keras.engine.saving import model_from_json
from keras.utils import multi_gpu_model
from keras import backend as K
from keras import optimizers
from keras_adabound import AdaBound
from keras_lookahead import Lookahead
from keras_radam import RAdam

from .callbacks import SaveMultiGPUModelCheckpoint
from .data_generator import DataGenerator
# from .utils import cou
from .utils import multi_threshold, recursive_glob, compute_class_weight, recursive_folder, config_dump
from .focal_loss import focal_loss
# from efficientnet_module.modules.config import Optimizer_, FREEZE
# from efficientnet_module.modules import config
from .callbacks import SGDLearningRateTracker

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
        self.id_class_mapping = None
        self.class_weights = None
        self.graph = tf.get_default_graph()
        self.session = K.get_session()
        self.config = config

    def _build_model(self):
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
            raise ValueError('Invalid EfficientNet architecture')

        # Redundant RGB channels for grayscale input
        base_model = model_class(input_shape=(self.input_size, self.input_size, 3),\
            weights= self.config.WEIGHT_PATH if self.config.WEIGHT_PATH is not None else 'imagenet',\
            include_top=False)

        freeze = None
        if freeze is None:
            pass
        else:
            for layer in base_model.layers:
                layer.trainable = not freeze

        x = keras.layers.GlobalAveragePooling2D()(base_model.output)
        output = keras.layers.Dense(self.num_of_classes, activation='softmax')(x)
        
        return keras.models.Model(inputs=[base_model.input], outputs=[output])

    def prepare_data(self):
        # TODO: Merge with Front-end and dev
        train_dir = os.path.join(self.config.DATASET_PATH, 'Train')
        train_dir = recursive_folder(train_dir)
        if self.config.AU_LIST:
            pass
        else:
            augment_dir = [x for x in train_dir if 'transformimage' in x.lower()]
            train_dir.remove(augment_dir[0])
        val_dir = os.path.join(self.config.DATASET_PATH, 'Validation')
        test_dir = os.path.join(self.config.DATASET_PATH, 'Test')

        # print(f"[DEBUG] Train dir:\n{train_dir}")
        # print(f"[DEBUG] Valdiation dir:\n{val_dir}")
        # print(f"[DEBUG] Test dir:\n{test_dir}")

        self.load_classes()

        self.train_generator = DataGenerator(train_dir, self.config.BATCH_SIZE,\
            self.classes, self.failClasses, self.passClasses,\
            self.binary_option, self.input_size, augmentation=self.config.AU_LIST)
        
        self.val_generator = DataGenerator(val_dir, self.config.BATCH_SIZE, \
            self.classes, self.failClasses, self.passClasses,\
            self.binary_option, self.input_size)
        self.test_generator = DataGenerator(test_dir, self.config.BATCH_SIZE, \
            self.classes, self.failClasses, self.passClasses,\
            self.binary_option, self.input_size)

        self.class_weights = compute_class_weight(self.train_generator.metadata.values())

    def lossFunc_chosen(self, option):
        try:
            loss_dict = {
                'binary': 'binary_crossentropy',
                'category' : 'categorical_crossentropy',
                'focal' : EfficientNetWrapper.loss_func(self.class_weights)
            }[option]
            return loss_dict
        except KeyError:
            raise ValueError("Invalid loss function")

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
                'nadam': optimizers.Nadam(),
                'adabound': AdaBound(),
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
        model.compile(optimizer=optimizer, loss=self.lossFunc_chosen('binary'), metrics=['accuracy'])
        model.fit_generator(self.train_generator, epochs=self.config.EPOCH, validation_data=self.val_generator, max_queue_size=10,
                            workers=1, callbacks=[checkpoint_callback, tensorboard_callback])

    def load_weight(self):
        self.keras_model = load_model(self.config.WEIGHT_PATH, compile=False)
        self.load_classes()

    def predict_one(self, img):
        resized_img = cv2.resize(img, (self.input_size, self.input_size))
        # resized_img = image_read(img, self.input_size)
        input_data = np.array([resized_img])
        X = preprocess_input(input_data)

        with self.graph.as_default():
            with self.session.as_default():
                Y = self.keras_model.predict(X)

        if self.config.CLASS_THRESHOLD is None or len(self.config.CLASS_THRESHOLD) == 0:
            Y_class_id = np.argmax(Y, axis=-1)
            Y_score = np.max(Y, axis=-1)
        else:
            ret = multi_threshold(Y, thresholds)
            if ret is None:
                classID = -1
                className = "Unknown"
                scores = Y[0]
                return classID, scores, className
            else:
                Y_class_id, Y_score = ret

        Y_class_name = self.id_class_mapping[Y_class_id[0]]
        return Y_class_id[0], Y_score[0], Y_class_name

    def evaluate(self, subset='test'):
        # TODO: make evaluate for all set instead of just test set
        optimizer = self.optimizer_chosen()
        self.keras_model.compile(optimizer=optimizer, loss=EfficientNetWrapper.loss_func(self.class_weights), metrics=['accuracy'])
        # self.keras_model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        subset_generator = {'train': self.train_generator, 'val': self.val_generator, 'test': self.test_generator}[
            subset]
        result = self.keras_model.evaluate_generator(subset_generator, verbose=1)
        print('Loss: %s' % result[0])
        print('Accuracy: %s' % result[1])

    def resume_training(self):
        epoch = 0

        if "startingmodel.h5" in self.config.WEIGHT_PATH:
            self.keras_model = self._build_model()
            self.load_classes()
        else:
            self.load_weight()

        # train
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.config.LOGS_PATH)
        
        if self.config.GPU_COUNT > 1:
            model = multi_gpu_model(self.keras_model, gpus=self.config.GPU_COUNT)
        elif self.config.GPU_COUNT == 1:
            model = self.keras_model
        else:
            raise ValueError("Invalid 'gpu_count' value")
        
        optimizer = self.optimizer_chosen()

        model.compile(optimizer=optimizer, loss=self.lossFunc_chosen('binary'), metrics=['accuracy'])
        # model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint_callback = SaveMultiGPUModelCheckpoint(self.keras_model, self.config.LOGS_PATH)
        model.fit_generator(self.train_generator, epochs=self.config.NO_EPOCH, validation_data=self.val_generator,
                            max_queue_size=10, workers=1, callbacks=[checkpoint_callback, tensorboard_callback],
                            initial_epoch=epoch)

    def confusion_matrix_evaluate(self):
        path =  [
            os.path.join(self.config.DATASET_PATH,"Train\\OriginImage"),    #Hardcode
            os.path.join(self.config.DATASET_PATH,"Validation"),            #Hardcode
            os.path.join(self.config.DATASET_PATH,"Test")                   #Hardcode
        ]
        for sub_path in path:
            print(f"[DEBUG] Evaluating: {sub_path}")
            image_list = []
            for image in glob.glob(sub_path + "/*.bmp"):
                image_list.append(image)
            # print(image_list)
            # break
            confusion_matrix = np.zeros(shape=(len(self.id_class_mapping) + 1, len(self.id_class_mapping) + 1), dtype=np.int)
            
            with tqdm.tqdm(total=len(image_list)) as pbar:
                for image_index, image_path in itertools.islice(enumerate(image_list), len(image_list)):

                    img, gt_name = load_and_crop(image_path, self.config.INPUT_SIZE)

                    if self.binary_option:

                        gt_name = 'Reject' if gt_name in self.failClasses else 'Pass'
                        gt_id = self.classes.index(gt_name)

                    else:

                        gt_id = self.classes.index(gt_name)
                    
                    pred_id, pred_score, pred_name = self.predict_one(img)

                    confusion_matrix[gt_id][pred_id] += 1

                    pbar.update()

            logger.info(f"Confusion matrix of {sub_path}")
            logger.info('\n%s' % confusion_matrix.astype(int))

    
    @staticmethod
    def loss_func(class_weight):
        # TODO: change loss function to be more dynamic
        def loss(y_true, y_pred):
            return focal_loss(y_true, y_pred, class_weight)
        
        return loss