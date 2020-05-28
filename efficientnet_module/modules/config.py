
# Size of the image
INPUT_SIZE = 192
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
EPOCH = 3
# Freeze Efficient option
FREEZE = False

# Model path
MODEL_DIR = 'models\\KLA_Infineon_model'

# Checkpoint path for training model
CHECKPOINT_PATH = 'KLA_Infinion_log_test'

# Epoch for test
TEST_EPOCH = 65


# Image path for train
DATASET_DIR = 'Dataset\\KLA_Infineon\\Dataset_KLA_fake_3'

# Image path for test
#cropped_folder\\Validation
#Dataset_yolo\\Validation
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\Dataset\\KLA_Infineon\\Dataset_KLA_fake_3\\Train\\OriginImage'
TEST_DIR = 'D:\\Coding_practice\\efficientnet\\Dataset\\KLA_Infineon\\KLA_Origin_data_same_illuminate\\Pass'
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\XRAY_dataset\\1_Lead'
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\XRAY_dataset\\2_Wire_short'
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\XRAY_dataset\\3_Foreign_material'
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\XRAY_dataset\\Pass'
# TEST_DIR = "D:\\Coding_practice\\efficientnet\\Skywork_dataset_compressed_nopadding\\Test"
# TEST_DIR = 'D:\\Coding_practice\\efficientnet\\Dataset_GF\\Test\\8_9_PATT_SMALL_DEF'
# TEST_DIR = "D:\\Coding_practice\\Skywork_data\\wafer_1\\Unclear_wafer1"
# TEST_DIR = "D:\\Coding_practice\\Skywork_data\\wafer_1\\Unknow_Defect_wafer1"
# TEST_DIR = "D:\\Coding_practice\\Skywork_data\\wafer 2 - Ifz\\Unclear_wafer2"
# TEST_DIR = "D:\\Coding_practice\\Skywork_data\\wafer 2 - Ifz\\Unknow_Defect_wafer2"
# TEST_DIR = "D:\\Coding_practice\\Skywork_data\\wafer 2 - Ifz\\Blur"
# TEST_DIR = "D:\\Coding_practice\\efficientnet\\Mis-classified\\Train"
# Architecture to use
ARCHITECTURE = 'B4'
"""
B0 - B7
"""
# Batch size
BATCH = 4

# GPU_count
GPU_COUNT = 1

# Optimizer
Optimizer_ = 'ranger'
""" Choose one of these optimizer
'sgd'
'adam'
'nadam'
'adabound'
'radam'
'ranger'
"""
