import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def normalize_image(image):
    if image is None:
        return image
    return image.astype('float32') / 255.0


def feature_extraction(img):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Definir un kernel para el filtro Laplaciano
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])
    
    # Aplicar el filtro Laplaciano a la imagen
    laplacian = cv2.filter2D(gray, -1, laplacian_kernel)
    
    # Aplicar la función de activación (ReLU) a la imagen
    relu = np.maximum(0, laplacian)
    
    # Devolver la imagen resultante
    return relu

#main 
if __name__ == '__main__':
    image_size = (128, 128)
    df_mask = pd.read_csv('../data/dataset/data/CASIA2/dataset.csv')
    #df_mask['Imagen_Original'] = df_mask['Imagen_Original'].apply(lambda x: cv2.imread(str(x)))
    df_mask['Imagen_Modificada'] = df_mask['Imagen_Modificada'].apply(lambda x: cv2.imread(str(x)))

    #df_mask['Imagen_Original'] = df_mask['Imagen_Original'].apply(lambda x: normalize_image(x))
    df_mask['Imagen_Modificada'] = df_mask['Imagen_Modificada'].apply(lambda x: normalize_image(x))

    df_mask['mascara'] = df_mask['mascara'].apply(lambda x: cv2.imread(str(x), cv2.IMREAD_GRAYSCALE))

    # eliminar filas con valores nulos o vacíos
    df_mask.dropna(inplace=True)
    # resize images to constant dimensions
    #df_mask['Imagen_Original'] = df_mask['Imagen_Original'].apply(lambda x: cv2.resize(x, image_size))
    df_mask['Imagen_Modificada'] = df_mask['Imagen_Modificada'].apply(lambda x: cv2.resize(x, image_size))

    df_mask['mascara'] = df_mask['mascara'].apply(lambda x: cv2.resize(x, image_size))

    imagen_modificada=df_mask['Imagen_Modificada'][10]
    mascara=df_mask['mascara'][10]
    #pass to color
    imagen_modificada=cv2.cvtColor(imagen_modificada, cv2.COLOR_BGR2RGB)
    mascara=cv2.cvtColor(mascara, cv2.COLOR_BGR2RGB)

    f = feature_extraction(imagen_modificada)

    #plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(imagen_modificada)
    ax[1].imshow(f, cmap='gray')
    plt.show()

    imagen_modificada=df_mask['Imagen_Modificada'][5]
    mascara=df_mask['mascara'][5]
    img_marcada = cv2.bitwise_and(imagen_modificada, imagen_modificada, mask=mascara)

    #crear copia de la imagen modificada
    imagen_modificada_= imagen_modificada.copy()

    # Enconrar los contornos de la máscara
    contours, hierarchy = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos en la imagen modificada
    cv2.drawContours(imagen_modificada, contours, -1, (0, 0, 255), 1)

    # combinar la imagen modificada con la imagen marcada usando addWeighted
    img_marcada = cv2.addWeighted(imagen_modificada, 0.5, img_marcada, 0.5, 0)

    # PONER A COLOR LAS IMAGENES
    imagen_modificada_  = cv2.cvtColor(imagen_modificada_, cv2.COLOR_BGR2RGB)

    img_marcada=cv2.cvtColor(img_marcada, cv2.COLOR_BGR2RGB)
    
    mascara = cv2.cvtColor(mascara, cv2.COLOR_BGR2RGB)

    print(imagen_modificada_.shape)
    print(mascara.shape)

    plt.figure(figsize=(16, 16))
    plt.subplot(131)
    plt.imshow(imagen_modificada_)
    plt.title('Imagen Modificada')
    plt.subplot(132)
    plt.imshow(mascara)
    plt.title('Máscara')
    plt.subplot(133)
    plt.imshow(img_marcada)
    plt.title('Imagen Marcada')
    plt.show()


    df_mask=df_mask[['Imagen_Modificada','mascara']]

    df_mask['Imagen_Modificada'] = df_mask['Imagen_Modificada'].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))
    df_mask['mascara'] = df_mask['mascara'].apply(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB))

    #dejar solo las imagenes modificadas y las mascaras
    df_mask=df_mask[['Imagen_Modificada','mascara']]

    # Entrenamos con 80% de los datos de train, usar 10% para validación y 10% para test
    X_train,X_val,y_train,y_val=train_test_split(df_mask['Imagen_Modificada'],df_mask['mascara'],test_size=0.2,random_state=42)
    X_val,X_test,y_val,y_test=train_test_split(X_val,y_val,test_size=0.5,random_state=42)

    #pasar a numpy array y redimensionar
    X_train=np.array(X_train.tolist())
    X_val=np.array(X_val.tolist())
    X_test=np.array(X_test.tolist())


    y_train=np.array(y_train.tolist())
    y_val=np.array(y_val.tolist())
    y_test=np.array(y_test.tolist())

    # ver dimensiones de los datos
    print(X_train.shape)
    print(y_train.shape)

    print(X_val.shape)
    print(y_val.shape)

    def build_model_mascara(X_train, Y_train, X_val, Y_val, img_size=(128,128)):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(img_size[0], img_size[1], 3)))
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Reshape((16, 16, 16)))
        model.add(Conv2DTranspose(filters=128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2DTranspose(filters=64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2DTranspose(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'))
        model.add(Conv2D(3, (1,1), activation='sigmoid', padding='same'))

        model.summary()

        epochs = 30
        batch_size = 32
        init_lr = 1e-4
        opt = Adam(lr=init_lr, decay=init_lr / epochs)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint('models/model_mascara.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val), callbacks=[early_stopping, checkpoint])

        # Plot training and validation accuracy and loss curves
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        #guardamos la imagen
        plt.savefig('accuracy_mascara.png')

        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        #guardamos la imagen
        plt.savefig('loss_mascara.png')

        #guardamos el modelo
        model.save('models/model_mascara.h5')
        return model

    # Entrenar el modelo
    model = build_model_mascara(X_train, y_train, X_val, y_val)

    # Generacion de Mascaras

import pandas as pd
import re

# Crear un diccionario para almacenar la información del dataset
dataset_dict = {'Imagen Original': [],
                'Imagen Modificada': [],
                'Máscara de Diferencias': [],
                'Área Modificada': []}

# Definir el tamaño de las imágenes
image_size = (512, 512)

# Definir las rutas de las imágenes
images_path_Au = '../data/dataset/data/CASIA2/Au/*.*'
images_path_Tp = '../data/dataset/data/CASIA2/Tp/*.*'

# Crear un diccionario para mapear los números de las imágenes a sus rutas
numImageToPath = {}
categories = {}
for file_path in glob.glob(images_path_Au):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    
    
    numImageAu = re.findall(r'\d+', os.path.splitext(os.path.basename(file_path))[0])[0]

    id = filename.split("_")[1] + numImageAu
    numImageToPath[id] = file_path


# Recorrer las imágenes y generar la máscara de diferencias
for file_path2 in glob.glob(images_path_Tp):
    
    filename2 = os.path.splitext(os.path.basename(file_path2))[0]

    numImageModificated = filename2.split("_")[5]

    numImageFromModification = filename2.split("_")[6]
    
    numIdImage = re.findall(r'\d+', filename2.split("_")[7])[0]

    if numImageModificated in numImageToPath and numImageFromModification in numImageToPath:
        if numImageModificated == numImageFromModification:
            original_image_path = numImageToPath[numImageFromModification]
            mask = generate_mask(original_image_path, file_path2, image_size)
            mask_path = f'../data/dataset/data/CASIA2/mask/{numImageFromModification}_{numImageModificated}_{numIdImage}.png'
        else:
            original_image_path = numImageToPath[numImageModificated]
            mask = generate_mask(original_image_path, file_path2, image_size)
            mask_path = f'../data/dataset/data/CASIA2/mask/{numImageModificated}_{numImageFromModification}_{numIdImage}.png'

    else:
        dataset_dict['Imagen Original'].append("NULL")
        dataset_dict['Imagen Modificada'].append(file_path2)
        dataset_dict['Máscara de Diferencias'].append("NULL")
        dataset_dict['Área Modificada'].append(0)

        continue
        
    # Guardar la máscara en la carpeta "data/dataset/data/CASIA2/mask"
    cv2.imwrite(mask_path, mask)
    quantity = np.count_nonzero(mask)

    # Agregar la información al diccionario del dataset
    dataset_dict['Imagen Original'].append(original_image_path)
    dataset_dict['Imagen Modificada'].append(file_path2)
    dataset_dict['Máscara de Diferencias'].append(mask_path)
    dataset_dict['Área Modificada'].append(quantity)

# Convertir el diccionario a un DataFrame de pandas y guardarlo en un archivo CSV
dataset_df = pd.DataFrame(dataset_dict)
dataset_df.to_csv('../data/dataset/data/CASIA2/mask/dataset.csv', index=False)
