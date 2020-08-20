import keras
import time
import tensorflow as tf

class SaveMultiGPUModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, template_model, checkpoint_dir):
        self.template_model = template_model
        self.checkpoint_dir = checkpoint_dir
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        self.template_model.save('%s/%s_%04d.h5' % (self.checkpoint_dir, time.strftime('%Y%m%d', time.localtime()), epoch))


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
        param = {
            "FN": 0,
            "TP": 0,
            "FP": 0,
            "TN": 0
        }
        print("====================================")
        print("Calculating FN/FP rate.....")
        print("====================================")
        for _, sample in enumerate(self.test_data):
            x_data, y_data = sample
            
            x_result = self.model.predict(x_data)

            predict_classes = x_result.argmax(axis=-1)

            for i in range(len(x_data)):
                # Check groundtruth
                y_current = y_data[i].tolist()

                if y_current.index(1) in self.fail_classes_index:
                    if predict_classes[i] == y_current.index(1)\
                        or predict_classes[i] in self.fail_classes_index:
                        param['TP'] += 1
                    else:
                        param['FN'] += 1
                else:
                    if predict_classes[i] == y_current.index(1)\
                        or predict_classes[i] in self.pass_classes_index:
                        param['TN'] += 1
                    else:
                        param['FP'] += 1

        FN_rate = (param['FN'] / (param['TP'] + param['FN'])) * 100
        FP_rate = (param['FP'] / (param['TN'] + param['FP'])) * 100

        print(f"Underkill rate: {FN_rate} %")
        print(f"Overkill rate: {FP_rate} %")

        items_to_write={
            "False Negative rate": FN_rate,
            "False Positive rate": FP_rate
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