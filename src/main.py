from utils import *
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


def split_data(dir_path, categories, train_prop=0.7, image_size=(64, 64)):
    # Obtener todas las imágenes auténticas y manipuladas
    auth_images = glob.glob(os.path.join(dir_path, "Au", "*.jpg"))
    manip_images = glob.glob(os.path.join(dir_path, "Tp", "*.jpg"))


    # Seleccionar aleatoriamente imágenes para los conjuntos de entrenamiento y prueba
    num_auth_train = round(len(auth_images) * train_prop)
    num_manip_train = round(len(manip_images) * train_prop)
    train_auth_indices = np.random.choice(len(auth_images), num_auth_train, replace=False)
    train_manip_indices = np.random.choice(len(manip_images), num_manip_train, replace=False)
    test_auth_indices = np.setdiff1d(np.arange(len(auth_images)), train_auth_indices)
    test_manip_indices = np.setdiff1d(np.arange(len(manip_images)), train_manip_indices)

    # Cargar solo las imágenes necesarias y redimensionarlas en paralelo
    train_images = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(resize_image)(auth_images[i], image_size) for i in train_auth_indices) + \
                   Parallel(n_jobs=multiprocessing.cpu_count())(delayed(resize_image)(manip_images[i], image_size) for i in train_manip_indices)
    
    # Agregar las máscaras de segmentación a las imágenes manipuladas
    manip_train_images = [manip_images[i] for i in train_manip_indices]
    masks = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(segment_image)(img_path) for img_path in manip_train_images)
    train_images += masks

    test_images = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(resize_image)(auth_images[i], image_size) for i in test_auth_indices) + \
                  Parallel(n_jobs=multiprocessing.cpu_count())(delayed(resize_image)(manip_images[i], image_size) for i in test_manip_indices)
    
    # Agregar las máscaras de segmentación a las imágenes manipuladas de prueba
    manip_test_images = [manip_images[i] for i in test_manip_indices]
    masks = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(segment_image)(img_path) for img_path in manip_test_images)
    test_images += masks
   
    # Crear etiquetas de entrenamiento y prueba
    train_labels = np.concatenate([np.zeros(num_auth_train), np.ones(num_manip_train)])
    test_labels = np.concatenate([np.zeros(len(auth_images) - num_auth_train), np.ones(len(manip_images) - num_manip_train)])


    # Agregar ceros a test_labels para igualar el número de elementos con test_images, incluyendo las máscaras de segmentación a la parte de prueba
    num_test_masks = len(masks)
    test_labels = np.concatenate([test_labels, np.zeros(num_test_masks)])

    #agregar ceros a train_labels para igualar el número de elementos con train_images, incluyendo las máscaras de segmentación a la parte de entrenamiento
    num_train_masks = len(train_images) - len(train_labels)
    train_labels = np.concatenate([train_labels, np.zeros(num_train_masks)])

    #Mostrar numero de elementos en cada clase de entrenamiento y prueba
    print("Número de elementos de train_images: ", len(train_images))
    print("Número de elementos de train_labels: ", len(train_labels))
    print("Número de elementos de test_images: ", len(test_images))
    print("Número de elementos de test_labels: ", len(test_labels))
    
    # Mezclar los datos de entrenamiento y prueba con scikit-learn
    train_images, test_images, train_labels, test_labels = train_test_split(np.concatenate([train_images, test_images]), np.concatenate([train_labels, test_labels]), test_size=1-train_prop, shuffle=True)

    return train_images, train_labels, test_images, test_labels


def resize_image(image_path, image_size):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, image_size, interpolation=cv2.INTER_AREA)
    return resized

def segment_image(image_path):
    # Leer imagen
    image = cv2.imread(image_path)

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar umbralización adaptativa
    thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    # Aplicar erosión y dilatación para eliminar el ruido y unir los objetos
    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # Aplicar detección de contornos
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en una imagen en blanco
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for contour in contours:
        cv2.drawContours(mask, [contour], -1, 255, -1)

    # Redimensionar la máscara a 64x64
    resized_mask = cv2.resize(mask, (64, 64))

    return resized_mask


def load_image(image_path):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    return image

def show_samples_images(dir_path, categories, sample_size=2):
    # Visualizar algunas imágenes de cada categoría del dataset
    sample_images = []
    sample_labels = []
    with ThreadPoolExecutor() as executor:
        for cat in categories:
            auth_image_path = glob.glob(os.path.join(dir_path, "Au", f"Au_{cat}_*.jpg"))[0]
            manip_image_path = glob.glob(os.path.join(dir_path, "Tp", f"Tp_D_CRN_S_N_{cat}*"))[0]
            image_paths = [auth_image_path, manip_image_path]
            images = list(executor.map(load_image, image_paths))
            sample_images.extend(images)
            sample_labels.extend([categories[cat], "Imagen Manipulada"])
    plt.figure(figsize=(15, 10))
    for i in range(sample_size * 2):
        plt.subplot(2, sample_size, i+1)
        plt.imshow(sample_images[i])
        plt.title(sample_labels[i])
    plt.show()


if __name__ == "__main__":
    # Iniciar el cronómetro
    start_time = time.time()

    # Ruta del dataset
    dir_path = os.path.join("data", "CASIA2.0", "CASIA2.0")

    # Hacer un diccionario con las categorías
    categories = {
        "ani": "Animales",
        "arc": "Arquitectura",
        "art": "Arte",
        "cha": "Personajes",
        "ind": "Interior",
        "nat": "Naturaleza",
        "pla": "Plantas",
        "txt": "Texturas"
    }

    # Patrones de archivo de imágenes auténticas y manipuladas
    authentic_pattern = os.path.join(dir_path, "Au", "*.jpg")
    manipulated_pattern = os.path.join(dir_path, "Tp", "*.jpg")

    # Contar el número de imágenes auténticas y manipuladas
    num_authentic = len(glob.glob(authentic_pattern))
    num_manipulated = len(glob.glob(manipulated_pattern))

    # Imprimir el número de imágenes auténticas y manipuladas
    print(f"Número de imágenes auténticas: {num_authentic}")
    print(f"Número de imágenes manipuladas: {num_manipulated}")

    # Proporción de entrenamiento y prueba
    train_prop = 0.7

    # Tamaño de las imágenes
    image_size = (224, 224)

    # Nombres de las clases
    class_of_image = ["Au", "Tp"]

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    train_images, train_labels, test_images, test_labels = split_data(
        dir_path,
        class_of_image,
        train_prop=train_prop,
        image_size=image_size
    )

    # Mostrar algunas imágenes de cada categoría del dataset
    show_samples_images(dir_path, categories)

    # Mostrar las imágenes de entrenamiento
    plot_images(train_images, train_labels, (10, 10))

    # Mostrar el tiempo de ejecución del programa
    print(f"--- {time.time() - start_time} seconds ---")



    
    

    


