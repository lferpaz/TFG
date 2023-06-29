import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps
from io import BytesIO
import pywt
import cv2
from scipy.fftpack import dct
from scipy import ndimage
from scipy import fftpack
import glob
import os

np.random.seed(2)

def ela_image(path, quality=98):
    temp_filename = 'temp_file_name.jpg'  # Nombre temporal para el archivo JPEG de la imagen
    ela_filename = 'temp_ela.png'  # Nombre de archivo para la imagen ELA
    
    image = Image.open(path).convert('RGB')  # Abrir la imagen y convertirla a modo RGB
    image.save(temp_filename, 'JPEG', quality=quality)  # Guardar la imagen como JPEG con la calidad especificada
    
    temp_image = Image.open(temp_filename)  # Abrir la imagen temporal
    ela_image = ImageChops.difference(image, temp_image)  # Obtener la imagen ELA calculando la diferencia de píxeles entre la imagen original y la imagen temporal
    
    extrema = ela_image.getextrema()  # Obtener los valores extremos de la imagen ELA
    max_diff = max([ex[1] for ex in extrema])  # Obtener el valor máximo de diferencia de los valores extremos
    if max_diff == 0:
        max_diff = 1  # Si la diferencia máxima es cero, se establece en uno para evitar divisiones por cero
    
    scale = 255.0 / max_diff  # Calcular la escala para ajustar el brillo de la imagen ELA
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)  # Ajustar el brillo de la imagen ELA según la escala calculada
    
    return ela_image 


def convert_to_ela_image(path, quality):
    with Image.open(path).convert('RGB') as image:  # Abrir la imagen y convertirla a modo RGB 
        with BytesIO() as image_bytes:  # Crear un objeto BytesIO para guardar la imagen temporalmente
            image.save(image_bytes, 'JPEG', quality=quality)  # Guardar la imagen en el objeto BytesIO como JPEG con la calidad especificada
            temp_image = Image.open(image_bytes)  # Abrir la imagen temporal desde el objeto BytesIO

            ela_image = ImageChops.difference(image, temp_image)  # Calcular la imagen ELA tomando la diferencia de píxeles entre la imagen original y la imagen temporal

            extrema = ela_image.getextrema()  # Obtener los valores extremos de la imagen ELA
            max_diff = max([ex[1] for ex in extrema])  # Obtener el valor máximo de diferencia de los valores extremos
            if max_diff == 0:
                max_diff = 1  # Si la diferencia máxima es cero, se establece en uno para evitar divisiones por cero

            scale = 255.0 / max_diff  # Calcular la escala para ajustar el brillo de la imagen ELA
            ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)  # Ajustar el brillo de la imagen ELA según la escala calculada

            return ela_image  
  

def discrete_wavelet_transform(image_path, size):
    # Cargar imagen y ajustar el tamaño
    img = Image.open(image_path)
    img = img.resize(size)
    # Convertir a escala de grises y luego a un arreglo numpy
    img = img.convert('L')
    img_array = np.array(img)
    # Aplicar la transformada wavelet discreta (DWT) usando la familia de wavelets 'haar'
    coeffs = pywt.dwt2(img_array, 'db20')
    # Extraer los coeficientes de nivel de detalle (high frequency)
    cA, (cH, cV, cD) = coeffs
    # Combinar los coeficientes de nivel de detalle para formar una sola imagen
    transformed_image = np.stack((cH, cV, cD), axis=2)
    # Normalizar los valores de la imagen para que estén en el rango [0, 255]
    transformed_image = (transformed_image - transformed_image.min()) * 255 / (transformed_image.max() - transformed_image.min())
    transformed_image = transformed_image.astype(np.uint8)
    # Devolver la imagen transformada
    return Image.fromarray(transformed_image)


def bdct(image_path):
    # Cargar imagen y ajustar el tamaño
    img = Image.open(image_path).convert('L')
    #img = img.resize(size)
    # Convertir a un arreglo numpy de punto flotante
    img_array = np.array(img).astype(np.float64)
    
    # Aplicar BDCT a cada bloque 8x8
    bdct = np.zeros_like(img_array)
    for i in range(0, img_array.shape[0], 8):
        for j in range(0, img_array.shape[1], 8):
            bdct[i:i+8, j:j+8] = dct(dct(img_array[i:i+8, j:j+8], axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Devolver los coeficientes transformados
    return Image.fromarray(bdct)


def decorrelate_image(image_path):
    # cargar la imagen y convertirla en una matriz de numpy
    image = Image.open(image_path)
    image_array = np.asarray(image)

    # reshape para obtener un arreglo 2D
    h, w, c = image_array.shape
    image_array = np.reshape(image_array, (h * w, c))

    # aplicar decorrelación
    cov = np.cov(image_array, rowvar=False)
    _, decorrelation_matrix = np.linalg.eigh(cov)
    transformed = np.dot(decorrelation_matrix, image_array.T).T

    # reshape para obtener la imagen resultante
    transformed_image = np.reshape(transformed, (h, w, c))

    # redimensionar a la salida deseada
    output_image = Image.fromarray(np.uint8(transformed_image))
    
    return output_image
    

def canny_image(image_path):
    # cargar la imagen y convertirla en escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # aplicar el operador Canny para detectar bordes
    canny = cv2.Canny(gray, 100, 200)

    # guardar la imagen resultante
    output_image = Image.fromarray(np.uint8(canny))
    return output_image



def enhance_features(image_path):
    # cargar la imagen y convertirla en una matriz de numpy
    image = Image.open(image_path)
    image_array = np.asarray(image)

    # aplicar filtro de bordes
    filtered_image = image.filter(ImageFilter.FIND_EDGES)

    # convertir la imagen filtrada en una matriz de numpy
    filtered_array = np.asarray(filtered_image)

    # aumentar el contraste de la imagen filtrada
    enhanced_array = filtered_array * 3

    # combinar la imagen original y la imagen filtrada para resaltar las características
    combined_array = np.maximum(image_array, enhanced_array)

    # crear y guardar la imagen resultante
    output_image = Image.fromarray(np.uint8(combined_array))
    

    return output_image


def highlight_features(image_path, factor=2):
    imagen = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para suavizar la imagen
    imagen_suave = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # Calcular la diferencia absoluta entre la imagen original y la suavizada
    imagen_diferencia = cv2.absdiff(imagen_gris, imagen_suave)

    # Escalar la diferencia para amplificar las características
    imagen_escala = cv2.convertScaleAbs(imagen_diferencia, alpha=factor)

    #normalizar la imagen
    imagen_escala = imagen_escala / 255

    return imagen_escala

def convert_to_sobel_image(imagen, img_size):
    # Convertir la imagen en escala de grises
    imagen = cv2.imread(imagen)

    imagen_gris = np.mean(imagen, axis=2)

    # redimensionar la imagen a la salida deseada
    imagen_gris = cv2.resize(imagen_gris, img_size)

    # Aplicar los filtros de Sobel para detectar bordes horizontales y verticales
    filtro_horizontal = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filtro_vertical = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    imagen_bordes_h = ndimage.convolve(imagen_gris, filtro_horizontal)
    imagen_bordes_v = ndimage.convolve(imagen_gris, filtro_vertical)

    # Calcular la magnitud de los bordes
    magnitud_bordes = np.sqrt(imagen_bordes_h ** 2 + imagen_bordes_v ** 2)

    # Normalizar la magnitud de los bordes para que estén en el rango [0, 255]
    magnitud_bordes *= 255.0 / np.max(magnitud_bordes)

    # Devolver la imagen resultante
    return magnitud_bordes



def plot_dwt2_from_file(path: str,img_size):
    # Load image
    original = plt.imread(path)
    original = cv2.resize(original, img_size)
    if original.ndim == 3:  # Convert to grayscale
        original = np.mean(original, axis=-1)

    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', 'Horizontal detail', 'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    plt.show()




################################### Enfoque con ROI ###################################

def detect_roi(img):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro gaussiano para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar un filtro Laplaciano para resaltar los bordes
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

    laplacian = np.uint8(np.absolute(laplacian))

    # Binarizar la imagen utilizando un umbral adaptativo
    thresh = cv2.adaptiveThreshold(laplacian, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Encontrar los contornos de la imagen binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande (posiblemente la ROI)
    max_area = 0
    best_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            best_contour = contour

    # Si se encontró un contorno, extraer la ROI
    if best_contour is not None:
        x, y, w, h = cv2.boundingRect(best_contour)
        roi = img[y:y+h, x:x+w]
    else:
        roi = img
      
    # normalizar la ROI y devolverla
    roi = roi / 255.0
    return roi


def get_ycc_channels(image_path):
    # Load image and convert to YCbCr color space
    image = Image.open(image_path).convert('YCbCr')
    ycc_image = np.array(image)
    
    # Extract Y, Cb, and Cr channels
    y_channel = ycc_image[:, :, 0]
    cb_channel = ycc_image[:, :, 1]
    cr_channel = ycc_image[:, :, 2]
    
    # Create new Y, Cb, and Cr images
    y_image = Image.fromarray(y_channel, mode='L')
    cb_image = Image.fromarray(cb_channel, mode='L')
    cr_image = Image.fromarray(cr_channel, mode='L')
    
    return y_image, cb_image, cr_image


def analyze_noise_patterns(image_path):
    # Carga la imagen y la convierte a una matriz NumPy
    with Image.open(image_path).convert('L') as image:
        image_data = np.array(image)

    # Realiza la transformada de Fourier en la imagen
    fft_data = fftpack.fft2(image_data)

    # Calcula la magnitud de la transformada de Fourier
    magnitude = np.abs(fft_data)

    # Calcula el logaritmo de la magnitud para visualizar mejor los patrones de ruido
    log_magnitude = np.log10(1 + magnitude)

    # Normaliza la matriz de log_magnitude para que sus valores estén entre 0 y 255
    log_magnitude_norm = (255*log_magnitude/log_magnitude.max()).astype(np.uint8)

    # Crea una imagen PIL a partir de la matriz normalizada
    noise_pattern_image = Image.fromarray(log_magnitude_norm)

    # Devuelve la imagen resultante
    return noise_pattern_image


################################### Preparete data ###################################

def preparete_analyze_noise_patterns(image_path, image_size):
    return np.array(analyze_noise_patterns(image_path).resize(image_size)).flatten() / 255.0

def preparete_image(image_path, image_size):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def preparete_image_ela(image_path, image_size = (128, 128)):
    return np.array(ela_image(image_path, 98).resize(image_size)).flatten() / 255.0

def preparete_highlightst_features(image_path, image_size):
    return np.array(highlight_features(image_path).resize(image_size)).flatten() / 255.0

def preparete_image_sovel(image_path, image_size):
    return np.array(convert_to_sobel_image(image_path, image_size)).flatten() / 255.0

def preparete_image_bdct(image_path, image_size):
    return np.array(bdct(image_path).resize(image_size)).flatten() / 255.0

def preparete_image_decorrelate(image_path, image_size):
    return np.array(decorrelate_image(image_path).resize(image_size)).flatten() / 255.0

def preparete_image_enhance(image_path, image_size):
    return np.array(enhance_features(image_path).resize(image_size)).flatten() / 255.0

def preparete_image_canny(image_path, image_size):
    return np.array(canny_image(image_path).resize(image_size)).flatten() / 255.0

