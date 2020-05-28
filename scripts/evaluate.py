import os

import sys


current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, MODEL_DIR, DATASET_DIR, TEST_EPOCH, ARCHITECTURE, BATCH

def main(model_architect_path, epoch, dataset_dir):
    model = EfficientNetWrapper(INPUT_SIZE)
    model.prepare_data(dataset_dir, BATCH)
    model.load_weight(model_architect_path, epoch, ARCHITECTURE, have_weight_only=True)
    model.evaluate('test')


if __name__ == '__main__':
    main(MODEL_DIR, TEST_EPOCH, DATASET_DIR)
