import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from PIL import Image
import cv2
import pandas as pd


def paste_object(img_original, boxes):
    '''
    Funcion para pegar el objeto en la imagen
    '''
    # Extraemos el objeto de la imagen original
    object = img_original.crop(boxes)

    # Pegamos el objeto en la imagen donde vamos a pegar el objeto, la posicion que sea random
    img_paste_copy = img_paste.copy()
    img_paste_copy.paste(object,(random.randint(0, img_paste_copy.size[0]), random.randint(0, img_paste_copy.size[1])))

    return img_paste_copy


def get_image_boxes(boxes_file):
    '''
    Funcion para leer las cajas que contiene el objeto
    '''
    # Leemos el fichero de texto donde se encuentran las cajas
    with open(boxes_file, 'r') as f:
        boxes = f.readlines()
    # Convertimos las cajas en una lista de listas y no colocamos la posicion 0 ya que es el nombre de la imagen
    boxes = [list(map(int, box.split('\t')[1:])) for box in boxes]
    return boxes


def show_image(img):
    '''
    Funcion para mostrar la imagen cargada con PIL
    '''
    plt.figure()
    plt.imshow(img)
    plt.show()


def copy_object(source_img, source_boxes, dest_img):
    # Se copian las cajas delimitadoras de origen para asegurar que el objeto se recorta correctamente
    x_min, y_min, x_max, y_max = source_boxes
    
    # Se recorta el objeto de la imagen de origen
    obj = source_img.crop((x_min, y_min, x_max, y_max))

    # Reescalamos el objeto para que sea como máximo del 50% de la imagen de destino y como mínimo del 10% de la imagen de destino. 
    # El porcentaje se elige aleatoriamente
    obj = obj.resize((random.randint(int(dest_img.width * 0.1), int(dest_img.width * 0.5)), 
                            random.randint(int(dest_img.height * 0.1), int(dest_img.height * 0.5))))
    
    # Se establece la región de destino como el tamaño del objeto
    dest_x = random.randint(0, dest_img.width - obj.width)
    dest_y = random.randint(0, dest_img.height - obj.height)

    # Se pega el objeto en la imagen de destino
    dest_img.paste(obj, (dest_x, dest_y))

    #devolvemos la imagen, la region de destino donde ha sido pegado el objeto 
    boxes = [dest_x, dest_y, dest_x + obj.width, dest_y + obj.height]
    
    return dest_img, boxes
    
def compare_images(img1,img2,img3):
    '''
    Funcion para comparar 3 imagenes, la original, la copiada y la pegada
    '''
    plt.figure()
    plt.title("Imagen original, copiada y pegada")
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.show()


if __name__ == '__main__':
    print('ImageNet dataset')
    # Ruta donde se encuentran las imagenes originales
    original_path = 'data/dataset/ImageNet/tiny-imagenet-200/tiny-imagenet-200/train/'

    #Crear dataset con las siguientes columnas: Nombre de la imagen original, coordenadas del objeto, imagen donde se ha pegado el objeto, coordenadas donde se ha pegado el objeto.
    dataset = pd.DataFrame(columns=['image_original','image_paste', 'boxes_paste'])


    #Hacer para todas las carpetas
    for folder in os.listdir(original_path):
        print("Generando imagenes para la carpeta: ", folder)

        # Ruta donde se encuentran los archivos con las cajas de los objetos a extraer
        boxes_path = 'data/dataset/ImageNet/tiny-imagenet-200/tiny-imagenet-200/train/'+folder+'/'+folder+'_boxes.txt'

        # Ruta donde se van a guardar las imagenes generadas
        generated_path = 'data/dataset/ImageNet/tiny-imagenet-200-copiado-pegado/' + folder + '/'

        # Creamos el directorio si no existe
        if not os.path.exists(generated_path):
            os.makedirs(generated_path)

        # Cargamos las imagenes originales
        images = glob.glob(original_path + folder + '/images/*.jpeg')

        original_imgs = []
        copy_imgs = []
        boxes_objects = []
        
        for img_file in images:
            # Leemos la imagen original
            img_original = Image.open(img_file)

            # Seleccionamos una imagen de forma aleatoria
            img_paste_file = random.choice(images)
            img_paste = Image.open(img_paste_file)

            # Obtenemos las cajas donde se encuentran los objetos de la imagen original
            boxes_ = get_image_boxes(boxes_path)
            

            #Obtenemos el indice de la imagen original
            index = img_file.split('_')[-1].split('.')[0]

            # Pegamos el objeto en la imagen donde vamos a pegar el objeto
            img_modificada, boxes = copy_object(img_original, boxes_[int(index)], img_paste)
            

            # Guardamos la imagen el nombre de la imagen donde se ha pegado el objeto + _CP.jpeg
            img_modificada.save(generated_path + folder + '_' + str(index) + '_CP.jpeg')

            #Guardamos la ruta de la imagen original, la imagen pegada y las cajas donde se encuentra el objeto
            original_imgs.append(img_file)
            copy_imgs.append(img_paste_file)
            boxes_objects.append(boxes)
            

        

        #agregamos los datos al dataset
        dataset = dataset.append(pd.DataFrame({'image_original':original_imgs, 'image_paste':copy_imgs, 'boxes_paste':boxes_objects}), ignore_index=True)

        print("Imagenes generadas para la carpeta: ----> ", folder)

    #Guardamos el dataset
    dataset.to_csv('data/dataset/ImageNet/tiny-imagenet-200-copiado-pegado/dataset.csv', index=False)

    
    



        


      

        

        


















'''import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import random
from PIL import Image


def paste_object(img_original,img_paste,boxes):

    Funcion para pegar el objeto en la imagen
    
    # Extraemos el objeto de la imagen original
    object = extract_object(img_original,boxes)

    # Pegamos el objeto en la imagen donde vamos a pegar el objeto, la posicion que sea random
    img_paste = img_paste.paste(object,(random.randint(0, img_paste.size[0]), random.randint(0, img_paste.size[1])))

    return img_paste

def extract_object(img,boxes):
    
    Funcion para extraer el objeto de la imagen
    
    return img.crop(boxes)

def get_image_boxes(boxes_file):
    
    Funcion para leer las cajas que contiene el objeto
    
    # Leemos el fichero de texto donde se encuentran las cajas
    with open(boxes_file, 'r') as f:
        boxes = f.readlines()
    # Convertimos las cajas en una lista de listas y no colocamos la posicion 0 ya que es el nombre de la imagen
    boxes = [box.split('\t')[1:] for box in boxes]
    # Convertimos las cajas a enteros
    boxes = [[int(box[0]), int(box[1]), int(box[2]), int(box[3])] for box in boxes]
    return boxes

def show_image(img):
    
    Funcion para mostrar la imagen cargada con PIL
    
    plt.figure()
    plt.imshow(img)
    plt.show()


def compare_images(img1,img2,img3):
    
    Funcion para comparar 3 imagenes, la original, la copiada y la pegada
    
    plt.figure()
    plt.title("Imagen original, copiada y pegada")
    plt.subplot(1,3,1)
    plt.imshow(img1)
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.subplot(1,3,3)
    plt.imshow(img3)
    plt.show()


def copy_object(source_img, source_boxes, dest_img):
    # Se copian las cajas delimitadoras de origen para asegurar que el objeto se recorta correctamente
    x_min, y_min, x_max, y_max = source_boxes
    
    # Se recorta el objeto de la imagen de origen
    obj = source_img[int(y_min):int(y_max), int(x_min):int(x_max), :]

    # Reescalamos el objeto para que sea como máximo del 50% de la imagen de destino y como mínimo del 10% de la imagen de destino. 
    # El porcentaje se elige aleatoriamente
    obj = cv2.resize(obj, (random.randint(int(dest_img.shape[1] * 0.1), int(dest_img.shape[1] * 0.5)), 
                            random.randint(int(dest_img.shape[0] * 0.1), int(dest_img.shape[0] * 0.5))))
    
    # Se establece la región de destino como el tamaño del objeto
    dest_x_min = 0
    dest_y_min = 0
    dest_x_max = obj.shape[1]
    dest_y_max = obj.shape[0]
    
    # Se eligen aleatoriamente las coordenadas de la esquina superior izquierda de la región de destino
    dest_h, dest_w, _ = dest_img.shape
    dest_x_min = random.randint(0, dest_w - obj.shape[1])
    dest_y_min = random.randint(0, dest_h - obj.shape[0])
    dest_x_max = dest_x_min + obj.shape[1]
    dest_y_max = dest_y_min + obj.shape[0]
    
    # Se pega el objeto recortado en la imagen de destino
    dest_img[dest_y_min:dest_y_max, dest_x_min:dest_x_max, :] = obj
    
    return dest_img


if __name__ == '__main__':
    # Cargamos la imagen original
    path_original = 'data/dataset/ImageNet/tiny-imagenet-200/tiny-imagenet-200/train/n01443537/images/n01443537_0.JPEG'
    path_modificada = 'data/dataset/ImageNet/tiny-imagenet-200/tiny-imagenet-200/train/n01443537/images/n01443537_1.JPEG'

    path_boxes = 'data/dataset/ImageNet/tiny-imagenet-200/tiny-imagenet-200/train/n01443537/n01443537_boxes.txt'

    boxes = get_image_boxes(path_boxes)

    # Cargamos la imagen original
    img_original = cv2.imread(path_original)
    # Cargamos la imagen donde vamos a pegar el objeto
    img_paste = cv2.imread(path_modificada)
    img_paste_copy = img_paste.copy()

    #Obtenemos el indice de la imagen que vamos a copiar a partir del path ya que el indice es el numero en el que termina la imagen _0.JPEG
    index = path_original.split('_')[-1].split('.')[0]

    copy_object(img_original,boxes[int(index)],img_paste)

    compare_images(img_original,img_paste_copy,img_paste)'''



   



    
