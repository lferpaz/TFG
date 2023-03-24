import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from skimage import color, util, io, img_as_float
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



def preprocess_images(images):
    """
    Realiza el preprocesamiento de una lista de imágenes.
    """
    preprocessed_images = []
    for image in images:
        # Convertir la imagen a YCbCr
        ycbcr_image = color.rgb2ycbcr(image)
        # Dividir la imagen en bloques de 8x8
        blocks = util.view_as_blocks(ycbcr_image[:, :, 0], block_shape=(8, 8))
        # Aplicar la transformada de coseno discreta en cada bloque
        dct_blocks = fftpack.dct(fftpack.dct(blocks, axis=2, norm='ortho'), axis=3, norm='ortho')
        # Agregar la imagen preprocesada a la lista
        preprocessed_images.append(dct_blocks)
    return preprocessed_images


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
