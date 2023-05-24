import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns

from scipy import fftpack

def read_folders(path):
    '''
    A partir del path que se nos pasa, leemos los diferentes folders que hay dentro y guardamos sus nombres en una lista
    '''
    folders = []
    for folder in os.listdir(path):
        folders.append(path + "/" + folder)
    return folders

def read_images(path):
    '''
    A partir del path que se nos pasa, guardamos las imagenes en una lista
    '''
    images = []
    for image in glob.glob(path):
        images.append(image)
    return images
    
def read_images(path):
    '''
    A partir del path que se nos pasa, guardamos todas la imagenes en una lista
    '''
    
    images = []
    for image in glob.glob(path):
        images.append(cv2.imread(image))
    return images
    


def convert_imgs_to_YCrCb(images):
    '''
    Funcion para convertir una lista de imagenes a YCrCb
    '''
    images_YCrCb = []
    for image in images:
        images_YCrCb.append(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))
    return images_YCrCb


def plot_processed_images(processed_images, n, figsize=(20,20)):
    """
    Función para plotear n imágenes preprocesadas en el mismo plot.
    """
    plt.figure(figsize=figsize)
    for i in range(n):
        # Obtener el primer bloque de la imagen preprocesada y aplicar la inversa de la transformada de coseno discreta
        # para reconstruir una imagen de 8x8 píxeles
        block = processed_images[i][0, 0]
        idct_block = fftpack.idct(fftpack.idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
        # Escalar los valores de la imagen para que estén en el rango [0, 255]
        scaled_block = (idct_block - np.min(idct_block)) * 255 / (np.max(idct_block) - np.min(idct_block))
        # Crear una imagen RGB con el bloque escalado en el canal Y y ceros en los canales Cb y Cr
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 0] = scaled_block.astype(np.uint8)
        # Mostrar la imagen
        plt.subplot(1,n,i+1)
        plt.imshow(image)
    plt.show()


def plot_image(image):
    '''
    Funcion para plotear una imagen
    '''
    plt.imshow(image)
    plt.show()


def plot_images(images, n, figsize=(20,20)):
    '''
    Funcion para plotear n imagenes en el mismo plot
    '''
    fig, axs = plt.subplots(1, n, figsize=figsize)
    for i in range(n):
        axs[i].imshow(images[i])
        axs[i].axis("off")
    plt.show()

def show_ycc_channels(y_image, cb_image, cr_image):
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    # Display Y, Cb, and Cr images in subplots
    axs[0].imshow(y_image, cmap='gray')
    axs[0].set_title('Y Channel')
    axs[1].imshow(cb_image, cmap='gray')
    axs[1].set_title('Cb Channel')
    axs[2].imshow(cr_image, cmap='gray')
    axs[2].set_title('Cr Channel')

    # Adjust subplot spacing
    fig.tight_layout()

    # Show the plot
    plt.show()

def show_cm(cm):
    # Representamos la matriz de confusión
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def detectar_alteraciones(img):
    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de Canny para detectar bordes
    edges = cv2.Canny(img_gray, 100, 200)

    # Encontrar contornos en la imagen de bordes
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear una imagen vacía para resaltar las áreas alteradas
    alteraciones = np.zeros(img.shape, np.uint8)

    # Iterar a través de los contornos y resaltar las áreas de la imagen que parecen estar alteradas
    for cnt in contours:
        # Calcular el área del contorno
        area = cv2.contourArea(cnt)

        # Ignorar los contornos que son demasiado pequeños o demasiado grandes
        if area < 50 or area > 500:
            continue

        # Dibujar el contorno en la imagen de las alteraciones
        cv2.drawContours(alteraciones, [cnt], 0, (0, 0, 255), 2)

    return alteraciones


def generate_mask(original_image_path, manipulated_image_path, size, reverse_order=False):
    # Leer las imágenes originales y manipuladas

    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    manipulated_image = cv2.imread(manipulated_image_path, cv2.IMREAD_GRAYSCALE)
   
    # Redimensionar las imágenes al tamaño especificado
    original_image = cv2.resize(original_image, size)
    manipulated_image = cv2.resize(manipulated_image, size)

    # Calcular la máscara de diferencias
    mask = cv2.absdiff(original_image, manipulated_image)

    # Binarizar la máscara para obtener una imagen en blanco y negro
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    # Aplicar un primer filtro de apertura para eliminar los pequeños detalles blancos
    kernel1 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)

    # Aplicar un segundo filtro de apertura para reducir aún más el número de puntos blancos
    kernel2 = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    # Devolver la máscara
    return mask
