from random import shuffle
import os
from efficientnet.keras import preprocess_input
from keras.utils import Sequence
import numpy as np
import cv2
# from .augmentation import function_augmentation
from .utils import to_onehot, crop_image_with_json
import json

class DataGenerator(Sequence):
    def __init__(self, input_dir, batch_size, num_of_classes, classes, input_size, augmentation=None):
    # def __init__(self, input_dir, batch_size, classes, input_size, label_smoothing=0., augmentation=None, mode=None):
        self.batch_size = batch_size
        self.input_size = input_size
        # self.mode = mode
        self.augmentation = augmentation
        # self.label_smoothing = label_smoothing
        self.input_dir = input_dir
        self.num_of_classes = num_of_classes
        self.classes = classes
        # self.classes_weights = np.zeros((1, num_of_classes))
        self.img_path_labels = self.load_data()

        if augmentation is None:
            self.augmentation = lambda x: x
        else:
            self.augmentation = lambda x: augmentation(images=x)

    def load_data(self):
        list_path_data = [self.input_dir]
        img_path_labels = []
        for path_data in list_path_data:
            for file in os.listdir(path_data):
                if file.split('.')[-1] == 'json':
                    with open(os.path.join(path_data, file), encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        ID = json_data['classId'][0]
                    img_path_labels.append((os.path.join(path_data,file.split('.json')[0]), to_onehot(self.classes.index(ID), self.num_of_classes)))
        return img_path_labels

    def __len__(self):
        return int(np.ceil(len(self.img_path_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.img_path_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [sample[0] for sample in batch]
        imgs = []
        for img_path in batch_x:
            # img = cv2.imread(img_path)
            img = crop_image_with_json(img_path+".json", img_path, self.input_size)
            
            img = cv2.resize(img, (self.input_size, self.input_size))
            imgs.append(img)
        batch_x = np.stack(imgs, axis=0)

        batch_x = self.augmentation(batch_x)
        batch_x = preprocess_input(batch_x)

        batch_y = [sample[1] for sample in batch]
        batch_y = np.array(batch_y)

        return batch_x, batch_y
    def on_epoch_end(self):
        shuffle(self.img_path_labels)