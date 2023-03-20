import datetime
import os
import cv2
import random
import time 
import glob
import numpy as np
import seaborn as sns
import multiprocessing
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard


def create_resnet50(input_shape):
    # Cargar el modelo ResNet50 pre-entrenado sin la capa final
    model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    # Congelar los pesos de todas las capas del modelo ResNet50
    for layer in model.layers:
        layer.trainable = False

    # Agregar una capa de Global Average Pooling para reducir la dimensionalidad de las características
    x = model.output
    x = GlobalAveragePooling2D()(x)

    # Agregar una capa totalmente conectada de 128 neuronas
    x = Dense(128, activation='relu')(x)

    # Agregar la capa de salida con una única neurona y función de activación sigmoidea
    predictions = Dense(1, activation='sigmoid')(x)

    # Combinar el modelo base con las nuevas capas
    model = Model(inputs=model.input, outputs=predictions)

    return model


def load_and_preprocess_image(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img /= 255.0
    return img

def load_and_preprocess_mask(img_path, image_size):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img /= 255.0
    img = tf.image.rgb_to_grayscale(img)
    return img


def split_data(dir_path, categories, train_prop=0.7, image_size=(64, 64), batch_size=32):
    # Obtener todas las imágenes auténticas y manipuladas
    auth_images = tf.data.Dataset.list_files(os.path.join(dir_path, "Au", "*.jpg"))
    manip_images = tf.data.Dataset.list_files(os.path.join(dir_path, "Tp", "*.jpg"))

    # Seleccionar aleatoriamente imágenes para los conjuntos de entrenamiento y prueba
    num_auth_images = len(list(auth_images))
    num_manip_images = len(list(manip_images))
    num_auth_train = round(num_auth_images * train_prop)
    num_manip_train = round(num_manip_images * train_prop)

    auth_images = auth_images.shuffle(buffer_size=num_auth_images, reshuffle_each_iteration=True)
    manip_images = manip_images.shuffle(buffer_size=num_manip_images, reshuffle_each_iteration=True)

    train_auth_images = auth_images.take(num_auth_train)
    train_manip_images = manip_images.take(num_manip_train)

    test_auth_images = auth_images.skip(num_auth_train)
    test_manip_images = manip_images.skip(num_manip_train)

    # Cargar imágenes de entrenamiento
    train_auth_images = train_auth_images.map(lambda x: load_and_preprocess_image(x, image_size))
    train_manip_images = train_manip_images.map(lambda x: load_and_preprocess_image(x, image_size))

    # Concatenar las imágenes auténticas y manipuladas
    train_images = train_auth_images.concatenate(train_manip_images)
    train_labels = tf.data.Dataset.from_tensor_slices(np.concatenate((np.zeros(num_auth_train), np.ones(num_manip_train))))

    # Mezclar los datos y crear lotes
    train_data = tf.data.Dataset.zip((train_images, train_labels)).shuffle(buffer_size=num_auth_train + num_manip_train).batch(batch_size)

    # Cargar imágenes de prueba
    test_auth_images = test_auth_images.map(lambda x: load_and_preprocess_image(x, image_size))
    test_manip_images = test_manip_images.map(lambda x: load_and_preprocess_image(x, image_size))

    # Concatenar las imágenes auténticas y manipuladas
    test_images = test_auth_images.concatenate(test_manip_images)
    test_labels = tf.data.Dataset.from_tensor_slices(np.concatenate((np.zeros(num_auth_images - num_auth_train), np.ones(num_manip_images - num_manip_train))))

    # Mezclar los datos y crear lotes
    test_data = tf.data.Dataset.zip((test_images, test_labels)).shuffle(buffer_size=num_auth_images + num_manip_images - num_auth_train - num_manip_train).batch(batch_size)

    return train_data, test_data


def train_model(train_data, test_data, epochs, batch_size, model_save_path):
    # Crear el modelo y compilarlo
    model = create_resnet50()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Crear una instancia del objeto TensorBoard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    # Entrenar el modelo
    history = model.fit(train_data, epochs=epochs, batch_size=batch_size, validation_data=test_data, callbacks=[tensorboard])

    # Guardar el modelo
    model.save(model_save_path)

    return history


if __name__ == "__main__":
    # Definir los parámetros
    dir_path = "data\CASIA2.0\CASIA2.0"
    # Definir las categorías de autenticidad
    categories = ["Au", "Tp"]

    # Definir la proporción de imágenes para el conjunto de entrenamiento
    train_prop = 0.7

    # Definir el tamaño de las imágenes de entrada
    image_size = (64, 64)

    # Definir el tamaño del lote
    batch_size = 32

    # Definir el número de épocas
    epochs = 10

    #Cargar y dividir los datos
    train_data, val_data = split_data(dir_path, categories, train_prop, image_size)

    imput_shape = ()
    #Definir el modelo
    model = create_resnet50((64,64,3))

    #Compilar el modelo
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #Definir los callbacks
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Entrenar el modelo 
    history = model.fit(train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=[tensorboard_callback])

    loss, acc = model.evaluate(val_data)

    print("Loss: ", loss)
    print("Accuracy: ", acc)

    









    

