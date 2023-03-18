import os
import glob
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

from utils import *

def split_data(dir_path, categories, train_prop=0.7, image_size=(224, 224)):
    # Obtener todas las imágenes auténticas y manipuladas
    auth_images = glob.glob(os.path.join(dir_path, "Au", "*.jpg"))
    manip_images = glob.glob(os.path.join(dir_path, "Tp", "*.jpg"))

    # Obtener las cantidades totales de imágenes auténticas y manipuladas
    num_auth = len(auth_images)
    num_manip = len(manip_images)

    # Obtener las cantidades de imágenes de entrenamiento y prueba para autenticidad y manipulación
    num_auth_train = round(num_auth * train_prop)
    num_manip_train = round(num_manip * train_prop)
    num_auth_test = num_auth - num_auth_train
    num_manip_test = num_manip - num_manip_train

    # Asegurarse de que cada conjunto tenga suficientes imágenes auténticas y manipuladas
    if num_auth_train + num_manip_train > 0 and num_auth_test + num_manip_test > 0:
        # Seleccionar aleatoriamente imágenes de autenticidad y manipulación para los conjuntos de entrenamiento y prueba
        auth_train = np.random.choice(auth_images, num_auth_train, replace=False)
        manip_train = np.random.choice(manip_images, num_manip_train, replace=False)
        auth_test = np.setdiff1d(auth_images, auth_train)
        manip_test = np.setdiff1d(manip_images, manip_train)

        # Redimensionar las imágenes de entrenamiento y prueba
        train_images = []
        for image_path in np.concatenate([auth_train, manip_train]):
            image = resize_image(image_path, image_size)
            train_images.append(image)
        train_images = np.array(train_images)
        test_images = []
        for image_path in np.concatenate([auth_test, manip_test]):
            image = resize_image(image_path, image_size)
            test_images.append(image)
        test_images = np.array(test_images)

        # Crear etiquetas de entrenamiento y prueba
        train_labels = np.concatenate([np.zeros(num_auth_train), np.ones(num_manip_train)])
        test_labels = np.concatenate([np.zeros(num_auth_test), np.ones(num_manip_test)])

        # Mezclar los datos de entrenamiento
        train_indices = np.arange(len(train_labels))
        np.random.shuffle(train_indices)
        train_images = train_images[train_indices]
        train_labels = train_labels[train_indices]

        # Mezclar los datos de prueba
        test_indices = np.arange(len(test_labels))
        np.random.shuffle(test_indices)
        test_images = test_images[test_indices]
        test_labels = test_labels[test_indices]

        return train_images, train_labels, test_images, test_labels
    else:
        print("No hay suficientes imágenes para crear conjuntos de entrenamiento y prueba.")


def show_samples_images(dir_path,categories, sample_size=5):
    # Visualizar algunas imágenes de cada categoría del dataset
    sample_images = []
    sample_labels = []
    for cat in categories:
        image_path = glob.glob(os.path.join(dir_path, "Au", f"Au_{cat}_*.jpg"))[0]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        sample_images.append(image)
        sample_labels.append(categories[cat])
        image_path = glob.glob(os.path.join(dir_path, "Tp", f"Tp_D_CRN_S_N_{cat}*"))[0]
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        sample_images.append(image)
        sample_labels.append("Imagen Manipulada")
    plt.figure(figsize=(15, 10))
    for i in range(sample_size * 2):
        plt.subplot(2, sample_size, i+1)
        plt.imshow(sample_images[i])
        plt.title(sample_labels[i])
    plt.show()



if __name__ == "__main__":
    # Ruta del dataset
    dir_path = "data/CASIA2.0/CASIA2.0"
    # Hacer un diccionario con las categorías
    categories = {"ani": "Animales", "arc": "Arquitectura", "art": "Arte", "cha": "Personajes", "ind": "Interior", "nat": "Naturaleza", "pla": "Plantas", "txt": "Texturas"}
    # Contar el número de imágenes auténticas y manipuladas
    num_authentic = len(glob.glob(os.path.join(dir_path, "Au", "*.jpg")))
    num_manipulated = len(glob.glob(os.path.join(dir_path, "Tp", "*.jpg")))
    print(f"Número de imágenes auténticas: {num_authentic}")
    print(f"Número de imágenes manipuladas: {num_manipulated}")
  
    show_samples_images(dir_path,categories)

    #Establecer la semilla aleatoria para reproducibilidad
    np.random.seed(42)

    # Definir los nombres de las clases
    class_of_image = ["Au", "Tp"]

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    train_images, train_labels, test_images, test_labels = split_data(dir_path, class_of_image,train_prop=0.7,image_size=(224, 224))

    # Imprimir la forma de las matrices de imágenes de entrenamiento y prueba
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(train_images[i])
        plt.title(f"Etiqueta: {train_labels[i]}")
        plt.axis("off")
    plt.show()


    
    

    


