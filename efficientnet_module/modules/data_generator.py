from random import shuffle

from efficientnet.keras import preprocess_input
from keras.utils import Sequence
from .augmentation_setup import simple_augmentation, skywork_augmentation, complex_augmentation
import numpy as np
import cv2
import glob
import json
import os
from .utils import to_onehot, load_and_crop, metadata_count


class DataGenerator(Sequence):
    def __init__(self, input_dir, batch_size, classes, failClasses, passClasses, input_size, binary_option, crop=True, augmentation=None):
        if isinstance(input_dir, list):
            self.input_dir = input_dir
        else:
            self.input_dir = [input_dir]

        self.failClasses = failClasses
        self.passClasses = passClasses
        self.binary_option = binary_option

        self.batch_size = batch_size
        self.classes = self.load_classes(classes)
        self.input_size = input_size
        self.crop = crop
        self.num_of_classes = len(self.classes)
        self.img_path_labels = self.load_data()
        self.metadata = metadata_count(self.input_dir, self.classes, self.gt_list, show_table=True)
        if augmentation:                # True: using simple augment
            self.augmentation = lambda x: x
        elif augmentation == None:      # augment untouch for validation and test data generator 
            self.augmentation = lambda x: x
        else:                           # False: using YOUR augment
            self.augmentation = lambda x: skywork_augmentation(images=x)

    def load_classes(self, classes):
        if self.binary_option:
            return ['Reject', 'Pass']
        else:
            return classes  

    # Change load_data if use other format
    def load_data(self):
        img_path_labels = []
        self.gt_list = []
        for path_data in self.input_dir:
            for img_path in glob.glob(os.path.join(path_data,"*.bmp")):
                json_path = img_path + ".json"
                # print(f"[DEBUG] {json_path}")
                try:
                    with open(json_path, encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        # print("[DEBUG] Json opened")
                    if self.binary_option:
                        id_image = 'Reject' if json_data['classId'][0] in self.failClasses else 'Pass'
                        # print(f'[DEBUG] {id_image}')
                    else:
                        id_image = json_data['classId'][0]
                    self.gt_list.append(id_image)
                    img_path_labels.append( (img_path, to_onehot(self.classes.index(id_image), self.num_of_classes)) )
                except:
                    print(f"[DEBUG] Missing {json_path}")
        # print(f"[DEBUG] {img_path_labels}")
        return img_path_labels
            
    def __len__(self):
        return int(np.ceil(len(self.img_path_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.img_path_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [sample[0] for sample in batch]
        imgs = []
        for img_path in batch_x:
            # img = image_read(img_path , self.input_size)
            if self.crop:
                img, _ = load_and_crop(img_path, self.input_size)
            else:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

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
