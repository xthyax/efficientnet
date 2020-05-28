import os
import sys
os.environ["PATH"] += os.pathsep +  "D:\\graphviz\\bin"
from imgaug import augmenters as iaa


current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from efficientnet_module.modules.augmentation_setup import complex_augmentation
from efficientnet_module.modules.config import INPUT_SIZE, DATASET_DIR, GPU_COUNT, ARCHITECTURE, BATCH, Optimizer_, CHECKPOINT_PATH
from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.utils import count_class
from efficientnet_module.modules import config as C_

# CHECKPOINT_PATH = 'logs'


def main(dataset_dir, input_size, gpu_count, checkpoint_path, last_epoch, efficientnet_architecture, batch_size, train_optimizer, continue_training, augmentation=None):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model = EfficientNetWrapper(input_size)
    dataset_path = os.path.join(current_dir,C_.DATASET_DIR) if C_.DATASET_DIR in os.listdir(current_dir) else C_.DATASET_DIR
    
    for directory in [d for d in (os.path.join(dataset_path, d1) for d1 in os.listdir(dataset_path)) if os.path.isdir(d)]:
        count_class(directory)
    _init_t =  input("Init train ?(Y/N)\nYour answer: ")
    if _init_t.lower() == "y":
        model.prepare_data(dataset_dir, batch_size= batch_size, augmentation=augmentation)
        if continue_training:
            model.resume_training(checkpoint_path, last_epoch, gpu_count=gpu_count, architecture=efficientnet_architecture)
        else:
            model.train(checkpoint_path, gpu_count= gpu_count, architecture=efficientnet_architecture)
    else:
        pass


if __name__ == '__main__':
    continue_training = False
    last_epoch = 'last'
    main(DATASET_DIR, INPUT_SIZE, GPU_COUNT, CHECKPOINT_PATH, last_epoch, ARCHITECTURE, BATCH, Optimizer_, continue_training , augmentation=complex_augmentation)
