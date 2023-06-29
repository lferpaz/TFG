import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam,SGD, Adagrad, RMSprop
from keras.utils import to_categorical
import os
import glob
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.layers import Dropout,BatchNormalization
from sklearn.model_selection import KFold
from tensorflow import keras


# Cargar el conjunto de datos CASIA V2.0
X = []
y = []


def test_model_with_external_images(model_path, image_paths):
    # Cargar el modelo entrenado
    model = keras.models.load_model(model_path)

    #Leer el contenido de la carpeta de imágenes usiando glob
    for file_path in glob.glob(image_paths+"/*"):
        img = cv2.imread(file_path)
        img_original = img.copy()
        if img is not None:
            # Antes de agregar la imagen la ponemos todas en el mismo tamaño
            img = cv2.resize(img, (128, 128))
            # Procesar la imagen
            X_processed = np.array([extract_features(threshold(preprocess(img)))])
            # Realizar la predicción
            prediction = np.argmax(model.predict(X_processed))

            # Mostrar la imagen con un texto que indica si es auténtica o modificada
            if prediction == 0:
                text = "Autentica"
            else:
                text = "Modificada"

            #cambiar el tamaño de la imagen para que se vea mas grande
            cv2.putText(img_original, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Imagen", img_original)
            cv2.waitKey(0)

    cv2.destroyAllWindows()




# Preprocesar las imágenes utilizando decorrelación y BDCT
def preprocess(X):
    X_gray = np.mean(X, axis=-1)
    X_dct = cv2.dct(X_gray.astype(np.float32))
    X_dct = np.expand_dims(X_dct, axis=-1)  # Agregar dimensión adicional
    X_dct[:, :, 1:] *= 0.1
    X_dct = cv2.idct(X_dct)
    X_dct = cv2.normalize(X_dct, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return X_dct

# Aplicar el método mejorado de umbral
def threshold(X):
    _, X_thresh = cv2.threshold(X, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return X_thresh

# Canales Cr, Y y Cb
def split_channels(X):
    if X.ndim == 2:
        # Si la imagen es en escala de grises, convertirla a formato BGR
        X = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
    elif X.ndim != 3 or X.shape[-1] != 3:
        raise ValueError("La imagen debe estar en formato BGR")

    # Convertir la imagen a formato YCrCb y separar los canales
    X_yuv = cv2.cvtColor(X, cv2.COLOR_BGR2YUV)
    X_cr, X_y, X_cb = cv2.split(X_yuv)
    return X_cr, X_y, X_cb

# Caracteristicas  Cr, Y y Cb para la clasificación
def extract_features(X):
    X_cr, X_y, X_cb = split_channels(X)
    X_features = np.stack([X_cr, X_y, X_cb], axis=-1)
    return X_features


def model(X_train, y_train, X_val, y_val, X_test, y_test):
    # Convertir las listas X e y en matrices numpy
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int32)
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int32)

    # Preprocesar y extraer características de las imágenes de entrenamiento y validación
    print("Preprocesar y extraer características de las imágenes de entrenamiento")
    X_train_processed = np.array([extract_features(threshold(preprocess(img))) for img in X_train])
    print("Preprocesar y extraer características de las imágenes de validación")
    X_val_processed = np.array([extract_features(threshold(preprocess(img))) for img in X_val])
    print("Preprocesar y extraer características de las imágenes de prueba")
    X_test_processed = np.array([extract_features(threshold(preprocess(img))) for img in X_test])
    
    # Convertir etiquetas a one-hot encoding
    print("Convertir etiquetas a one-hot encoding")
    y_train_onehot = to_categorical(y_train, num_classes=2)
    y_val_onehot = to_categorical(y_val, num_classes=2)
    y_test_onehot = to_categorical(y_test, num_classes=2)
    
    # Crear modelo CNN
    print ("Crear modelo CNN")
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    
    # Compilar modelo
    optimizer = Adam(lr=0.0001) # Learning rate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
    # Implementar Early stopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    
    # Entrenar modelo
    print("Entrenar modelo")
    model.fit(X_train_processed, y_train_onehot, epochs=10, batch_size=32, verbose=1,
              validation_data=(X_val_processed, y_val_onehot), callbacks=[early_stopping])
    
    # Evaluar modelo con los datos de validación
    print("Evaluar modelo con los datos de validación")
    val_loss, val_acc = model.evaluate(X_val_processed, y_val_onehot, verbose=0)
    print("Pérdida de validación:", val_loss)
    print("Precisión de validación:", val_acc)

    # Evaluar modelo con los datos de prueba
    print("Evaluar modelo con los datos de prueba")
    test_loss, test_acc = model.evaluate(X_test_processed, y_test_onehot, verbose=0)
    print("Pérdida de prueba:", test_loss)
    print("Precisión de prueba:", test_acc)

    # Guardar modelo
    print("Guardar modelo")
    model.save('model'+str(val_acc)+'.h5')

    return model


if __name__ == '__main__':

    imput_shape = (244,244)
    # Cargar las imágenes auténticas 
    for file_path in glob.glob('data/dataset/data/CASIA2/Au/*.jpg') [0:6000]:
        img = cv2.imread(file_path)
        if img is not None:
            #antes de agregar la imagen la ponemos todas en el mismo tamaño
            img = cv2.resize(img, (256, 256))
            X.append(img)
            y.append(0)

    # Cargar las imágenes falsas 
    for file_path in glob.glob('data/dataset/data/CASIA2/Tp/*.jpg') [0:6000]:
        img = cv2.imread(file_path)
        if img is not None:
            #antes de agregar la imagen la ponemos todas en el mismo tamaño
            img = cv2.resize(img, (256, 256))
            X.append(img)
            y.append(1)


    # Dividir los datos en conjunto de entrenamiento (70%), validación (15%) y prueba (15%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.18, random_state=42)

    # Entrenar el modelo
    model = model(X_train, y_train, X_val, y_val, X_test, y_test)

    #Test del modelo
    #test_model_with_external_images('model0.7904525728456293.h5', 'tests')
