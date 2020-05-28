import os
import sys
from pathlib import Path

import cv2

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.utils import recursive_glob
from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, MODEL_DIR, TEST_EPOCH, TEST_DIR

def main(model_dir, epoch, test_dir, thresholds, input_size):
    model = EfficientNetWrapper(input_size)
    model.load_weight(model_dir, epoch)

    img_paths = recursive_glob(test_dir, '*.bmp')
    for img_path in img_paths:
        img = cv2.imread(img_path)
        prediction = model.predict_one(img, thresholds)
        print('Pred: %s - %s' % (prediction[2], prediction[1]))
        cv2.imshow(Path(img_path).name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    thresholds = [0.62, 0.38, 0.38, 0.38]
    main(MODEL_DIR, TEST_EPOCH, TEST_DIR, thresholds, INPUT_SIZE)
