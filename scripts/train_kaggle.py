import os
import sys
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)
    
from efficientnet_module.modules.augmentation_setup import complex_augmentation, simple_augmentation
from efficientnet_module.modules.efficientnet_wrapper import EfficientNetWrapper
from efficientnet_module.modules.utils import count_class
from efficientnet_module.modules.config_kaggle import INPUT_SIZE, DATASET_DIR, ARCHITECTURE, BATCH, Optimizer_, CHECKPOINT_PATH, FREEZE, IMAGE_PATH
from efficientnet_module.modules import config_kaggle
from efficientnet_module.modules.kaggle_data_generator import DataGenerator
from efficientnet_module.modules.utils import config_dump
from efficientnet_module.modules.callbacks import SaveMultiGPUModelCheckpoint
import pandas as pd
import keras
from keras.callbacks import ModelCheckpoint
from efficientnet.keras import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def split_train_test(dataframe, classes):
    labels_list = []
    sample_list = []
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    for class_ in classes:
        value = np.where(dataframe[class_].values == 1)[0]
        
        labels = []
        samples = []
        for index in value:
            label = dataframe.iloc[index,1:].values
            sample = dataframe.iloc[index,0]
            
            labels.append(label)
            samples.append(sample)
        Sub_X_train, Sub_X_val, Sub_y_train, Sub_y_val = train_test_split(samples, labels, train_size= 0.8, random_state=2)

        X_train.extend(Sub_X_train)
        y_train.extend(Sub_y_train)
        X_val.extend(Sub_X_val)
        y_val.extend(Sub_y_val)

    return X_train, y_train, X_val, y_val
   
if __name__ == '__main__':
    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    df = pd.read_csv(DATASET_DIR + "train.csv")

    train_labels = df.loc[:,'healthy':'scab'].values
    train_id = df.loc[:,'image_id'].values
    
    classes = df.columns.values[1:]
    # X_train, y_train, X_val, y_val = split_train_test(df, classes)
    X_train, X_val, y_train, y_val = train_test_split(train_id, train_labels, train_size=0.8, random_state=2, shuffle=True)

    print(np.sum(y_train,axis=0))
    print(np.sum(y_val,axis=0))
    input("Ready ?")
# Prepare data

    train_generator = DataGenerator(IMAGE_PATH, BATCH, len(classes), X_train, y_train, INPUT_SIZE, augmentation=simple_augmentation)

    val_generator = DataGenerator(IMAGE_PATH, BATCH, len(classes), X_val, y_val, INPUT_SIZE)

    y_decode = [np.where(r==1)[0][0] for r in y_train]

    # print(y_decode)
    class_weight = compute_class_weight('balanced', np.unique(y_decode), y_decode)
# # Build model
    model = EfficientNetWrapper(INPUT_SIZE)
    model.load_classes_kaggle(classes)
    optimizer =model.optimizer_chosen(Optimizer_)
    model = model._build_model(ARCHITECTURE, FREEZE)
    """
    base_model = EfficientNetB5(input_shape=(input_size, input_size, 3), weights='imagenet', include_top=False)
    # for layer in base_model.layers:
    #     layer.trainable=False
    x = keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = keras.layers.Dense(len(classes), activation='softmax')(x)

    model = keras.models.Model(inputs=[base_model.input], outputs=[output])
    """
    config_dump(CHECKPOINT_PATH, config_kaggle)
    
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=CHECKPOINT_PATH)
    # my_callback = ModelCheckpoint(CHECKPOINT_PATH + '/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_best_only=False, save_weights_only=False)
    checkpoint_callback = SaveMultiGPUModelCheckpoint(model, CHECKPOINT_PATH)

    model.fit_generator(train_generator, epochs=3000, validation_data=val_generator, max_queue_size=10,
                            workers=1, callbacks=[checkpoint_callback, tensorboard_callback], class_weight= class_weight)


