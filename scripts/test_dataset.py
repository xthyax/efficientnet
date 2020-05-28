import os
import sys

import cv2

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.config import INPUT_SIZE, MODEL_DIR, TEST_DIR, TEST_EPOCH

def main(model_dir, epoch, test_dir, thresholds):
    model = EfficientNetWrapper(INPUT_SIZE)
    model.load_weight(model_dir, epoch)

    with open('%s/labels.txt' % test_dir) as f:
        while True:
            img_name = f.readline().strip()
            class_name = f.readline().strip()
            img = cv2.imread('%s/%s' % (test_dir, img_name))
            prediction = model.predict_one(img, thresholds)
            print('GT: %s' % class_name)
            print('Pred: %s - %s' % (prediction[2], prediction[1]))
            cv2.imshow(img_name, img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == '__main__':
    thresholds = None
    main(MODEL_DIR, TEST_EPOCH, TEST_DIR, thresholds)
