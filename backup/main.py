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
import random

from sklearn.metrics import confusion_matrix
import seaborn as sns


'''physical_devices = tf.config.experimental.list_physical_devices('GPU')
#limitamos la memoria de la GPU para que no se quede sin memoria
tf.config.experimental.set_virtual_device_configuration(physical_devices[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
print("Number of GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)'''

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

   
    real_images_path = 'data/dataset/data/CASIA2/Au/*.*'
    fake_images_path = 'data/dataset/data/CASIA2/Tp/*.*'
    image_size = (128, 128)

    ###############################################  MODEL TO CLASIFICATE REAL/MANIPULATED IMAGES #########################################################
    X = []
    y = []

    print("Cargando datos de imágenes reales...")
    for file_path in glob.glob(real_images_path):
            X.append(preparete_image_ela(file_path, image_size))
            y.append(0)

    random.shuffle(X)

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

    
    model = build_model_2(X_train, Y_train, X_val, Y_val,image_size)
    #model = tf.keras.models.load_model('models/MODEL_V2_ELA_0.943_128.h5')
    
    # Calculamos la precisión del modelo con los datos de test
    score = model.evaluate(X_test, Y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    
    # Calcula la matriz de confusión
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_true=Y_test.argmax(axis=1), y_pred=y_pred.argmax(axis=1))

    
    

    '''class_names = ['real', 'fake']
    real_image = os.listdir('data/dataset/data/CASIA2/Au/')
    fake_images = os.listdir('data/dataset/data/CASIA2/Tp/')



    correct_r = 0
    total_r = 0
    for file_name in fake_images[0:100]:
        if file_name.endswith('jpg') or file_name.endswith('png'):
            real_image_path = os.path.join('data/dataset/data/CASIA2/Tp/', file_name)
            image = preparete_image_ela(real_image_path, image_size)
            image = image.reshape(-1, 128, 128, 3)
            y_pred = model.predict(image)
            y_pred_class = np.argmax(y_pred, axis = 1)[0]
            total_r += 1
            if y_pred_class == 1:
                correct_r += 1
                print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')

    print(f'Correct: {correct_r} Total: {total_r} Accuracy: {correct_r / total_r * 100:0.2f}')

    #Probamos el modelo cargando una imagen falsificada y una real
    img_path= 'data/dataset/data/CASIA2/Au/Au_ani_00029.jpg'
    img_path2= 'data/dataset/data/CASIA2/Tp/Tp_D_CND_S_N_txt00028_txt00006_10848.jpg'
    
    #Cargamos la imagen original en color
    img_original = cv2.imread(img_path)
   

    class_names = ['fake', 'real']
    image = preparete_image_ela(img_path, image_size)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    print(f'Class: {class_names[y_pred_class]} Confidence: {np.amax(y_pred) * 100:0.2f}')
    
    
    if y_pred_class == 0:
        cv2.putText(img_original, 'Manipulated', (img_original.shape[1] - 200, img_original.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(img_original, 'Original', (img_original.shape[1] - 200, img_original.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    #mostrar imagen con el resultado de la predicción
    cv2.imshow('Result', img_original)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''


    ###############################################  MODEL TO TAMPERED IMAGES  #########################################################

    '''print("Cargando datos de imágenes falsificadas...")
    X_Tp,Y_Tp= create_dataset_for_tampered_images('data/dataset/data/CASIA2/Tp')
   

    print("X shape: ", np.array(X_Tp).shape)
    print("y shape: ", np.array(Y_Tp).shape)

    X_Tp= np.array(X_Tp).reshape(-1,128,128,3)
    Y_Tp = to_categorical(Y_Tp, num_classes = 2)

    # Entrenamos con 80% de los datos de train, usar 10% para validación y 10% para test
    X_train, X_val, Y_train, Y_val = train_test_split(X_Tp, Y_Tp, test_size = 0.2, random_state=5)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.5, random_state=5)

    
    model = build_splice_classification_model(X_train, Y_train, X_val, Y_val)
    
    # Calculamos la precisión del modelo con los datos de test
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])'''

   







