import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

import tensorflow as tf

from preprocess import *
from model import *
from utils import *

from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_data():
    x = []
    y = []
    for folder in ['Au', 'Tp']:
            for filename in glob.glob(os.path.join('data/dataset/data/CASIA2/', folder, '*.jpg'))[0:7000]:
                x.append(preparete_image(filename))

                if folder == 'Tp':
                    y.append(1)
                else:
                    y.append(0)

    return x, y


def create_dataset_for_tampered_images(path_tampered):
    x = []
    y = []

    for filename in glob.glob(os.path.join(path_tampered, '*.*')):
         #Obtener del path 'data/dataset/data/CASIA2/Tp\\Tp_D_CND_S_N_txt00028_txt00006_10848.jpg' el nombre de la imagen 'Tp_D_CND_S_N_txt00028_txt00006_10848.jpg'
        img_name = filename.split('\\')[-1]
         #si la imagen empieza por TP_S_ es una imagen falsificada de manera copiar y pegar
        if  img_name.startswith('Tp_S_'):
            x.append(preparete_image(filename, (128, 128)))
            y.append(1)
        elif  img_name.startswith('Tp_D_'):
            x.append(preparete_image(filename, (128, 128)))
            y.append(0)

    return np.array(x), y


if __name__ == '__main__':

   
    real_images_path = 'data/dataset/data/CASIA2/Au/*.jpg'
    fake_images_path = 'data/dataset/data/CASIA2/Tp/*.jpg'
    image_size = (128, 128)

    ###############################################  MODEL TO CLASIFICATE REAL/MANIPULATED IMAGES #########################################################
    X = []
    y = []


    print("Cargando datos de imágenes reales...")
    for file_path in glob.glob(real_images_path):
            X.append(preparete_image_ela(file_path, image_size))
            y.append(0)
    print("Cargando datos de imágenes falsificadas...")
    for file_path in glob.glob(fake_images_path):
            X.append(preparete_image_ela(file_path, image_size))
            y.append(1)
    
    print("X shape: ", np.array(X).shape)
    print("y shape: ", np.array(y).shape)

    X= np.array(X).reshape(-1, image_size[0], image_size[1], 3)
    y = to_categorical(y, num_classes = 2)

    # Entrenamos con 80% de los datos de train, usar 10% para validación y 10% para test
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.2, random_state=5)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.5, random_state=5)

    
    model = build_model(X_train, Y_train, X_val, Y_val,image_size)
    
    # Calculamos la precisión del modelo con los datos de test
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Calcula la matriz de confusión
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true=Y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1))

    
   

   







