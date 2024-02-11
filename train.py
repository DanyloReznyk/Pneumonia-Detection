import tensorflow as tf
import numpy as np
import pandas as pd
import os
import pydicom

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import cv2
import gc
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import *
from PIL import Image 

from architectures import model1, model2, model3, model4
from constants import SEED, BATCH_SIZE, VAL_BATCH_SIZE, IMAGE_SIZE, ADJUSTED_IMAGE_SIZE, TRAIN_DF_PATH
from constants import DETAIL_DF_PATH, TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, TRAIN_SAMPLES, VALID_SAMPLES, EPOCH_COUNT
AUTOTUNE = tf.data.AUTOTUNE



INPUT_SHAPE = (ADJUSTED_IMAGE_SIZE, ADJUSTED_IMAGE_SIZE, 3)

def create_sets():
    labels  = pd.read_csv(TRAIN_DF_PATH)
    details = pd.read_csv(DETAIL_DF_PATH)
    details = details.drop_duplicates('patientId').reset_index(drop=True)
    labels_w_class = labels.merge(details, how='inner', on='patientId')
    labels_w_class.fillna(0, inplace=True)

    new_df = labels_w_class.head(6000)

    class_train, class_val = train_test_split(new_df, test_size = 0.20, random_state = SEED, stratify = new_df['class'])
 
    FACTOR = ADJUSTED_IMAGE_SIZE/IMAGE_SIZE

    train_images_dir = TRAIN_IMAGES_PATH
    def create_mask(datafm, n_classes = 2):
        X = []
        y = []

        masks = np.zeros((int(datafm.shape[0]), ADJUSTED_IMAGE_SIZE, ADJUSTED_IMAGE_SIZE)) #MASK_IMAGE_SIZE -> ADJUSTED_IMAGE_SIZE

        for index, patient_id in enumerate(datafm['patientId'].T.to_dict().values()):
            image_path = os.path.join(train_images_dir, patient_id) + ".dcm"
            img = pydicom.dcmread(image_path)
            img = img.pixel_array

            img = cv2.resize(img, (ADJUSTED_IMAGE_SIZE, ADJUSTED_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
            img = Image.fromarray(img)
            img = img.convert('RGB')
            img = preprocess_input(np.array(img, dtype=np.float32))

            X.append(img)
            rows = labels_w_class[labels_w_class['patientId']==patient_id]
            y.append(rows['Target'].values[0])

            row_data = list(rows.T.to_dict().values())
            for row in row_data:
                x1 = int(row['x']*FACTOR)
                x2 = int((row['x']*FACTOR)+(row['width']*FACTOR))
                y1 = int(row['y']*FACTOR)
                y2 = int((row['y']*FACTOR)+(row['height']*FACTOR))
                masks[index][y1:y2, x1:x2] = 1

        del img,row,row_data
        gc.collect()

        X= np.array(X)
        y= np.array(y)
        
        train_masks_input = np.expand_dims(masks, axis=3)
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))

        return X, y, y_train_cat, masks

    X_train, y_tr_target, y_train , mask_train = create_mask(class_train)
    X_val, y_val_target, y_val , mask_val  = create_mask(class_val)

    train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
    val = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(VAL_BATCH_SIZE)

    return train, val


def train_models(train, val, model_f, model_s, model_t, model_fo):

    print('Models compilation')
    model1 = model_f()
    model2 = model_s()
    model3 = model_t()
    model4 = model_fo()

    print('Models are fitting')

    h1 = model1.fit(train, validation_data = val, epochs=EPOCH_COUNT)
    h2 = model2.fit(train, validation_data = val, epochs=EPOCH_COUNT)
    h3 = model3.fit(train, validation_data = val, epochs=EPOCH_COUNT)
    h4 = model4.fit(train, validation_data = val, epochs=EPOCH_COUNT)

    print('Saving weights')
    model1.save_weights('checkpoints/model1.h5')
    model2.save_weights('checkpoints/model2.h5')
    model3.save_weights('checkpoints/model3.h5')
    model4.save_weights('checkpoints/model4.h5')

    print('All stages were successful.')

train_set, val = create_sets()
train_models(train_set, val, model1, model2, model3, model4)
