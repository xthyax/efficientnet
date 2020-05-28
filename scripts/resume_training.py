import os
import sys

from imgaug import augmenters as iaa

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet.modules.efficientnet_wrapper import EfficientNetWrapper


def main(dataset_dir, input_size, batch_size, gpu_count, checkpoint_path, last_epoch, augmentation=None):
    model = EfficientNetWrapper(input_size)
    model.prepare_data(dataset_dir, batch_size=batch_size, augmentation=augmentation)
    model.resume_training(checkpoint_path, last_epoch, gpu_count=gpu_count)


if __name__ == '__main__':
    dataset_dir = 'E:\data\pj\lensformer\Process\\20191230\classification_dataset'
    input_size = 300
    batch_size = 16
    gpu_count = 4
    last_epoch = 'last'
    checkpoint_path = 'data/classification'

    augmentation = iaa.Sequential([
        # iaa.CoarseDropout((0.001, 0.002), size_percent=0.03125),
        iaa.OneOf([
            iaa.Flipud(1),
            iaa.Affine(rotate=90),
            iaa.Affine(rotate=180),
            iaa.Affine(rotate=270),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Affine(rotate=90),
            ]),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Affine(rotate=180),
            ]),
            iaa.Sequential([
                iaa.Flipud(1),
                iaa.Affine(rotate=270),
            ])
        ])
    ])

    main(dataset_dir, input_size, batch_size, gpu_count, checkpoint_path, last_epoch, augmentation=augmentation)
