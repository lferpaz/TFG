import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)
from PIL import Image, ImageChops, ImageEnhance,ImageFilter
from io import BytesIO
import pywt
import cv2
from scipy.fftpack import dct
from skimage import color, io, util




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
        
def convert_to_wavelet_image(image_path, image_size):
    # Cargar la imagen y convertirla a escala de grises
    with Image.open(image_path) as image:
        gray_image = image.convert('L')
    
    # Aplicar el filtro bilateral para reducir el ruido
    filtered_image = np.asarray(gray_image.filter(ImageFilter.SMOOTH_MORE))
    
    # Aplicar la transformada de wavelet de Daubechies
    coeffs = pywt.dwt2(filtered_image, 'db2')
    cA, (cH, cV, cD) = coeffs
    
    # Concatenar los coeficientes de la transformada y aplanarlos
    features = np.concatenate([cA, cH, cV, cD]).flatten()
    
    # Redimensionar los coeficientes para que tengan el tamaño especificado
    resized_features = cv2.resize(features, image_size)

    return resized_features

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


def bdct(image_path, size):
    # Cargar imagen y ajustar el tamaño
    img = Image.open(image_path).convert('L')
    img = img.resize(size)
    # Convertir a un arreglo numpy de punto flotante
    img_array = np.array(img).astype(np.float64)
    
    # Aplicar BDCT a cada bloque 8x8
    bdct = np.zeros_like(img_array)
    for i in range(0, img_array.shape[0], 8):
        for j in range(0, img_array.shape[1], 8):
            bdct[i:i+8, j:j+8] = dct(dct(img_array[i:i+8, j:j+8], axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Devolver los coeficientes transformados
    return Image.fromarray(bdct)



def decorrelate_image(image_path, output_size):
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
    output_image = output_image.resize(output_size)

    return output_image
    

def highlight_image_features(image_path, output_size):
    # cargar la imagen y convertirla en escala de grises
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # aplicar el operador Canny para detectar bordes
    canny = cv2.Canny(gray, 100, 200)

    # redimensionar a la salida deseada
    resized = cv2.resize(canny, output_size)

    # guardar la imagen resultante
    output_image = Image.fromarray(resized)
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

def highlight_features(imagen, factor=20):
    """
    Esta función resalta las principales características de una imagen al calcular la diferencia absoluta entre
    la imagen original y una versión suavizada de la misma. Luego, aplica una función de escalado para amplificar
    las diferencias y devolver una imagen en escala de grises que resalta dichas características.

    Args:
    - imagen: numpy.ndarray. La imagen a procesar.
    - factor: float. El factor de escalado para amplificar las diferencias. Por defecto, se utiliza 20.

    Returns:
    - numpy.ndarray. La imagen procesada.
    """
    # Convertir la imagen a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Aplicar un filtro Gaussiano para suavizar la imagen
    imagen_suave = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

    # Calcular la diferencia absoluta entre la imagen original y la suavizada
    imagen_diferencia = cv2.absdiff(imagen_gris, imagen_suave)

    # Escalar la diferencia para amplificar las características
    imagen_escala = cv2.convertScaleAbs(imagen_diferencia, alpha=factor)

    return imagen_escala


def preparete_highlights_image(image_path, image_size):
    return np.array(highlight_image_features(image_path, image_size)).flatten() / 255.0


def preparete_image_wavelet(image_path, image_size):
    return np.array(convert_to_wavelet_image(image_path, image_size)).flatten() / 255.0


def preparete_image(image_path, image_size):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

def preparete_image_bdct(image_path, image_size):
    return np.array(bdct(image_path, 8).resize(image_size)).flatten() / 255.0

def preparete_image_decorrelate(image_path, image_size):
    return np.array(decorrelate_image(image_path, image_size)).flatten() / 255.0

def preparete_image_enhance(image_path, image_size):
    return np.array(enhance_features(image_path).resize(image_size)).flatten() / 255.0


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



