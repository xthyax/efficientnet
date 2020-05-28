import keras
import time

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