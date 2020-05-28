from random import shuffle
import os
from efficientnet.keras import preprocess_input
from keras.utils import Sequence
import numpy as np
import cv2
from .augmentation import function_augmentation
from .utils import to_onehot
import json

class DataGenerator(Sequence):
    def __init__(self, input_dir, batch_size, classes, input_size, label_smoothing=0., augmentation=None, mode=None):
        self.batch_size = batch_size
        self.input_size = input_size
        self.mode = mode
        self.augmentation = augmentation
        self.label_smoothing = label_smoothing
        self.input_dir = input_dir
        self.num_of_classes = len(classes)
        self.classes = np.array(classes)
        self.classes_weights = np.zeros((1, len(self.classes)))
        self.img_path_labels = self.load_data()
        if self.augmentation == "True":
            self.augmentation = lambda x: x
        else:
            self.augmentation = lambda x: function_augmentation(images=x)

    def load_data(self):
        if self.mode == 'Train':
            self.augmentation = None
            if self.augmentation == 'True':
                list_path_data= ['%s\\OriginImage' % self.input_dir, '%s\\TransformImage' % self.input_dir]
            else:
                list_path_data = ['%s\\OriginImage' % self.input_dir]
        else:
            list_path_data = [self.input_dir]
        img_path_labels = []
        for path_data in list_path_data:
            for file in os.listdir(path_data):
                if file.split('.')[-1] == 'json':
                    with open(os.path.join(path_data, file), encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        ID = json_data['classId'][0]
                        for ids, label in enumerate(self.classes):
                            if label == ID:
                                id_label = ids
                        if self.mode == 'Train':
                            self.classes_weights[0][id_label] +=1
                        img_path_labels.append((os.path.join(path_data,file.split('.json')[0]), to_onehot(int(id_label), self.num_of_classes, self.label_smoothing)))
        return img_path_labels

    def __len__(self):
        return int(np.ceil(len(self.img_path_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.img_path_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [sample[0] for sample in batch]
        imgs = []
        for img_path in batch_x:
            img = cv2.imread(img_path)

            def center_point(json_path, size_image):
                with open(json_path, encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    box = json_data['box']
                    startX = box['centerX'][0]
                    startY = box['centerY'][0]
                    widthBox = box['widthBox'][0]
                    heightBox = box['heightBox'][0]
                # center_x = int(startX) + int(widthBox/2)
                # center_y = int(startY) + int(heightBox/2)
                center_x = int(startX)
                center_y = int(startY)
                if center_x < self.input_size:
                    center_x = self.input_size
                elif center_x > size_image[0] - self.input_size:
                    center_x = size_image[0] - self.input_size
                if center_y < self.input_size:
                    center_y = self.input_size
                elif center_y >size_image[1]-self.input_size:
                    center_y = size_image[1]-self.input_size
                return int(center_x), int(center_y)
            center_x, center_y = center_point(img_path+".json", img.shape)
            
            img = img[center_y-int(self.input_size/2):center_y+int(self.input_size/2), center_x-int(self.input_size/2):center_x+int(self.input_size/2)]
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