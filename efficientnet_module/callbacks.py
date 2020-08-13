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

class SGDLearningRateTracker(keras.callbacks.Callback):
    def lr_calculate(self, optimizer):
        self.lr = optimizer.lr
        self.decay = optimizer.decay
        self.iterations = optimizer.iterations
        lr = self.lr
        lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        return lr
# class AdamLearningRateTracker(Callback):
class CustomCallback(keras.callbacks.Callback):
    def __init__(self, tb_callback, test_data, list_classes, fail_classes, pass_classes):
        self.tb_callback = tb_callback
        self.test_data = test_data
        self.classes = list_classes
        self.fail_classes_index = [list_classes.index(class_) for class_ in fail_classes]
        self.pass_classes_index = [list_classes.index(class_) for class_ in pass_classes]

    def on_epoch_end(self, epoch, logs=None):
        # Underkill rate := FN / (TP + FN)
        # Overkill rate := FP / (TN + FP)
        param = {
            "FN": 0,
            "TP": 0,
            "FP": 0,
            "TN": 0
        }
        print("====================================")
        print("Calculating Underkill/Overkill rate.....")
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

        Underkill_rate = param['FN'] / (param['TP'] + param['FN'])
        Overkill_rate = param['FP'] / (param['TN'] + param['FP'])

        print(f"Underkill rate: {Underkill_rate} %")
        print(f"Overkill rate: {Overkill_rate} %")

        items_to_write={
            "Underkill_rate": Underkill_rate,
            "Overkill_rate": Overkill_rate
        }
        writer = self.tb_callback.writer
        for name, value in items_to_write.items():
            summary = tf.compat.v1.summary.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name 
            writer.add_summary(summary, epoch)
            writer.flush()
