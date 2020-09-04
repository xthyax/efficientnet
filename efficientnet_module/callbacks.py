import keras
import time
import tensorflow as tf
import sys
import numpy as np
from datetime import datetime

class SaveMultiGPUModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, template_model, checkpoint_dir):
        self.template_model = template_model
        self.checkpoint_dir = checkpoint_dir
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.template_model.save('%s/%s_%04d.h5' % (self.checkpoint_dir, time.strftime('%Y%m%d', time.localtime()), epoch), include_optimizer=False)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, writer,test_data, list_classes, fail_classes, pass_classes):
        # self.tb_callback = tb_callback
        self.writer = writer
        self.test_data = test_data
        self.classes = list_classes
        self.fail_classes_index = [list_classes.index(class_) for class_ in fail_classes]
        self.pass_classes_index = [list_classes.index(class_) for class_ in pass_classes]

    def on_epoch_end(self, epoch, logs=None):
        # False Negative rate := FN / (TP + FN)
        # False Positive rate := FP / (TN + FP)
        print("====================================")
        print("Calculating FN/FP rate.....")
        print("====================================")

        y_gth_list = []
        y_pred_list = []

        for _, sample in enumerate(self.test_data):
            x_data, y_data = sample
            
            x_result = self.model.predict(x_data)

            gth_classes = np.argmax(y_data, axis=-1)
            predict_classes = x_result.argmax(axis=-1)

            y_gth_list.extend(gth_classes)
            y_pred_list.extend(predict_classes)

        fail_gth_list  = [np.array(y_gth_list) == class_ for class_ in self.fail_classes_index]
        fail_gth_list = np.sum(fail_gth_list, axis=0)
        total_fail = np.sum(fail_gth_list)
        false_fail_pred_list = [np.array(y_pred_list) == class_ for class_ in self.fail_classes_index]
        false_fail_pred_list = np.invert(np.sum(false_fail_pred_list, axis=0).astype('bool'))
        false_fail_pred_list = false_fail_pred_list * fail_gth_list
        total_underkill = np.sum(false_fail_pred_list)
        UK_rate = (total_underkill / total_fail) * 100

        pass_gth_list = [np.array(y_gth_list) == class_ for class_ in self.pass_classes_index]
        pass_gth_list = np.sum(pass_gth_list, axis=0)
        total_pass = np.sum(pass_gth_list)
        false_pass_pred_list = [np.array(y_pred_list) == class_ for class_ in self.pass_classes_index]
        false_pass_pred_list = np.invert(np.sum(false_pass_pred_list, axis=0).astype('bool'))
        false_pass_pred_list = false_pass_pred_list  * pass_gth_list
        total_overkill = np.sum(false_pass_pred_list)
        OK_rate = (total_overkill / total_pass ) * 100
        
        print(f"Underkill rate: {UK_rate} %")
        print(f"Overkill rate: {OK_rate} %")

        items_to_write={
            "Undekill rate": UK_rate,
            "Overkill rate": OK_rate
        }

        self.writer.add_scalars("Custom_metric",{list(items_to_write)[0]: items_to_write[list(items_to_write)[0]],\
                                                list(items_to_write)[1]: items_to_write[list(items_to_write)[1]]}, epoch)

        self.writer.flush()

        # writer = self.tb_callback.writer
        # for name, value in items_to_write.items():
        #     summary = tf.compat.v1.summary.Summary()
        #     summary_value = summary.value.add()
        #     summary_value.simple_value = value
        #     summary_value.tag = name 
        #     writer.add_summary(summary, epoch)
        #     writer.flush()

    def on_train_end(self, logs=None):
        self.writer.close()