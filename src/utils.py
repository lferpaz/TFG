import cv2

#Ejemplo de una funcion bien comentada
def resize_image(image_path, size):
    """
    Redimensiona la imagen en `image_path` al tamaño `size`.

    Parameters:
        image_path (str): El camino de la imagen a redimensionar.
        size (tuple): El tamaño deseado de la imagen redimensionada en la forma (width, height).

    Returns:
        La imagen redimensionada.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img
