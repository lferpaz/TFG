import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def resize_image(image_path, image_size):
    # Cargar la imagen
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    # Redimensionar la imagen
    image = cv2.resize(image, image_size)
    return image

def normalize_images(images):
    normalized_images = []
    for image in images:
        # Convertir la imagen a flotante y normalizar
        normalized_image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        # Añadir la imagen normalizada a la lista
        normalized_images.append(normalized_image)
    return normalized_images

def plot_images(images,labels,size):

    plt.figure(figsize=size)
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i])
        plt.title(f"Etiqueta: {labels[i]}")
        plt.axis("off")
    plt.show()

