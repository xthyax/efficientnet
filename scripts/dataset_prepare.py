import os
import sys
import tqdm
import cv2
import itertools
from prettytable import PrettyTable
import json

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
from efficientnet_module.modules import config as C_
from efficientnet_module.modules.utils import count_class
dash_path = "\\" if os.sys == "nt" else "/"
# print(os.listdir(current_dir))

dataset_path = os.path.join(current_dir,C_.DATASET_DIR) if C_.DATASET_DIR in os.listdir(current_dir) else C_.DATASET_DIR

# print([d for d in (os.path.join(dataset_path, d1) for d1 in os.listdir(dataset_path)) if os.path.isdir(d)])
# dI for dI in os.listdir('foo') if os.path.isdir(os.path.join('foo',dI))

for directory in [d for d in (os.path.join(dataset_path, d1) for d1 in os.listdir(dataset_path)) if os.path.isdir(d)]:
    count_class(directory)

