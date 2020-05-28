from random import shuffle

from efficientnet.keras import preprocess_input
from keras.utils import Sequence
import numpy as np
import cv2

from .utils import to_onehot, image_read


class DataGenerator(Sequence):
    def __init__(self, input_dir, batch_size, num_of_classes, input_size, augmentation=None):
        self.batch_size = batch_size
        self.input_size = input_size
        with open('%s/labels.txt' % input_dir) as f:
            labels = f.readlines()

        labels = [line.strip() for line in labels]
        self.img_path_labels = [tuple(labels[i:i + 2]) for i in range(0, len(labels), 2)]
        self.img_path_labels = [('%s/%s' % (input_dir, sample[0]), to_onehot(sample[1], num_of_classes)) for
                                sample in self.img_path_labels]

        if augmentation is None:
            self.augmentation = lambda x: x
        else:
            self.augmentation = lambda x: augmentation(images=x)

    def __len__(self):
        return int(np.ceil(len(self.img_path_labels) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch = self.img_path_labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = [sample[0] for sample in batch]
        imgs = []
        for img_path in batch_x:
            # img = image_read(img_path , self.input_size)
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
