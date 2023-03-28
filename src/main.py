import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from PIL import Image, ImageChops, ImageEnhance, ImageOps
import os
import itertools
from io import BytesIO
import glob
import random

def convert_to_ela_image(path, quality):
    with Image.open(path).convert('RGB') as image:
        with BytesIO() as image_bytes:
            image.save(image_bytes, 'JPEG', quality=quality)
            temp_image = Image.open(image_bytes)

            ela_image = ImageChops.difference(image, temp_image)

            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1
            scale = 255.0 / max_diff

            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

            return ela_image
        

def preparete_image(image_path, image_size):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0


def build_model(X_train, Y_train, X_val, Y_val):
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()

    epochs = 40
    batch_size = 32
    init_lr = 1e-4
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 2,verbose = 0,mode = 'auto')
    
    model.fit(X_train,Y_train,batch_size = batch_size, epochs = epochs,validation_data = (X_val, Y_val), callbacks = [early_stopping])
    
    #calculamos la precisión del modelo
    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    
    model.save('models/model'+str(score[1])+'.h5')

    return model


if __name__ == '__main__':
    real_images_path = 'data/dataset/data/CASIA2/Au/*.jpg'
    fake_images_path = 'data/dataset/data/CASIA2/Tp/*.jpg'
    image_size = (128, 128)

    X = []
    y = []

    for file_path in glob.glob(real_images_path)[:6000]:
            X.append(preparete_image(file_path, image_size))
            y.append(0)

    for file_path in glob.glob(fake_images_path)[:6000]:
            X.append(preparete_image(file_path, image_size))
            y.append(1)

    print("X shape: ", np.array(X).shape)
    print("y shape: ", np.array(y).shape)

    X= np.array(X).reshape(-1,128,128,3)
    y = to_categorical(y, num_classes = 2)

    #La idea es entrenar con 80% de los datos de train, usar 10% para validación y 10% para test
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.2, random_state=5)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.5, random_state=5)


    model = build_model(X_train, Y_train, X_val, Y_val)

    #calculamos la precisión del modelo con los datos de test
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
