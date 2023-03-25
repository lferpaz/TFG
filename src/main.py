import cv2
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
import os
import glob

# Cargar el conjunto de datos CASIA V2.0
X = []
y = []

# Cargar las imágenes auténticas y guardarlas en la lista X y la lista y con valor 0 limitado a 1000 imágenes
for file_path in glob.glob('data/dataset/data/CASIA2/Au/*.jpg') [0:2000]:
    img = cv2.imread(file_path)
    if img is not None:
        #antes de agregar la imagen la ponemos todas en el mismo tamaño
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(0)

# Cargar las imágenes falsas y guardarlas en la lista X y la lista y con valor 1
for file_path in glob.glob('data/dataset/data/CASIA2/Tp/*.jpg') [0:2000]:
    img = cv2.imread(file_path)
    if img is not None:
        #antes de agregar la imagen la ponemos todas en el mismo tamaño
        img = cv2.resize(img, (128, 128))
        X.append(img)
        y.append(1)

# Convertir las listas X e y en matrices numpy
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

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

# Separar los canales Cr, Y y Cb
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

# Utilizar las características resultantes de Cr, Y y Cb para la clasificación
def extract_features(X):
    X_cr, X_y, X_cb = split_channels(X)
    X_features = np.stack([X_cr, X_y, X_cb], axis=-1)
    return X_features

# Modelo CNN con kernel RBF y validación cruzada k-fold
kf = KFold(n_splits=5)
accuracies = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print("Preprocesar y extraer características de las imágenes")
    X_train_processed = np.array([extract_features(threshold(preprocess(img))) for img in X_train])
    X_test_processed = np.array([extract_features(threshold(preprocess(img))) for img in X_test])

    print("Convertir etiquetas a one-hot encoding")
    y_train_onehot = to_categorical(y_train, num_classes=2)
    y_test_onehot = to_categorical(y_test, num_classes=2)

    print ("Crear modelo CNN")
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compilar modelo
    optimizer = Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Entrenar modelo
    model.fit(X_train_processed, y_train_onehot, epochs=10, batch_size=32, verbose=1)

    # Evaluar modelo
    y_pred = np.argmax(model.predict(X_test_processed), axis=-1)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

#imprimir la precisión media
print('Accuracy: ', np.mean(accuracies))