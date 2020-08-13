
# Size of the image
INPUT_SIZE = 600
"""
EfficientNet B0 : 224
EfficientNet B1 : 240
EfficientNet B2 : 260
EfficientNet B3 : 300
EfficientNet B4 : 380
EfficientNet B5 : 456
EfficientNet B6 : 528
EfficientNet B7 : 600

Excel these size with corresponding architecture will reduce model accuracy
"""
EPOCH = 200
# Freeze Efficientnet option
FREEZE = True

# Model path
MODEL_DIR = ''

# Checkpoint path for training model
CHECKPOINT_PATH = "/mnt/sdc1/Projects/Pixel/jhyn/EfficientNet_modify/kaggle_log_test"

# Epoch for test
TEST_EPOCH = 6

# Image path for train
DATASET_DIR = "D:\\Coding_practice\\plan_pathology_kaggle\\"

IMAGE_PATH = "D:\\Coding_practice\\plan_pathology_kaggle\\images"
# Image path for test

TEST_DIR = 'D:\\Coding_practice\\plan_pathology_kaggle\\images'

# Architecture to use
ARCHITECTURE = 'B5'
"""
B0 - B7
"""
# Batch size
BATCH = 1

# GPU_count
GPU_COUNT = 1

# Optimizer
Optimizer_ = 'adam'
""" Choose one of these optimizer
'sgd'
'adam'
'nadam'
'adabound'
'radam'
'ranger'
"""