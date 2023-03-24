import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from glob import glob

# Función para preprocesar la imagen
def preprocess_img(img):
    img_resized = cv2.resize(img, (64, 64))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    if img_gray.dtype != np.float32:
        img_gray = img_gray.astype(np.float32)
    img_dct = cv2.dct(img_gray)
    return img_dct

# Función para extraer características de la imagen
def extract_features(img):
    if len(img.shape) == 3:
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    img = cv2.convertScaleAbs(img)
    # Threshold using Otsu's method
    cr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Extract features using Hu moments
    moments = cv2.HuMoments(cv2.moments(cr)).flatten()
    return moments

if __name__ == "__main__":
    # Cargar las imágenes de entrenamiento y prueba
    train_auth_path = "data/dataset/training/au/"
    train_manip_path = "data/dataset/training/tampered/"

    test_auth_path = "data/dataset/testing/au/"
    test_manip_path = "data/dataset/testing/tampered/"
    
    # Obtener las rutas de las imágenes de entrenamiento y prueba
    train_paths = glob(train_auth_path + "*.png") + glob(train_manip_path + "*.tif")
    test_paths = glob(test_auth_path + "*.png") + glob(test_manip_path + "*.tif")
    
    # Cargar las imágenes de entrenamiento y prueba
    train_imgs = [cv2.imread(img) for img in train_paths]
    test_imgs = [cv2.imread(img) for img in test_paths]

    # Preprocesar las imágenes de entrenamiento y prueba
    train_imgs_preprocessed = [preprocess_img(img) for img in train_imgs]
    test_imgs_preprocessed = [preprocess_img(img) for img in test_imgs]

    # Extraer características de las imágenes de entrenamiento y prueba
    train_features = np.array([extract_features(img) for img in train_imgs_preprocessed])
    test_features = np.array([extract_features(img) for img in test_imgs_preprocessed])

    # Definir las etiquetas de clasificación de las imágenes de entrenamiento y prueba
    train_labels = np.array([0] * len(glob(train_auth_path + "*.png")) + [1] * len(glob(train_manip_path + "*.tif")))
    test_labels = np.array([0] * len(glob(test_auth_path + "*.png")) + [1] * len(glob(test_manip_path + "*.tif")))


    # Definir el modelo SVM con kernel RBF
    svm = SVC(kernel='rbf')

    # Realizar la validación cruzada k-fold para evaluar el rendimiento del modelo
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(train_features):
        X_train, X_test = train_features[train_index], train_features[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    print("Precisión promedio en validación cruzada: {:.2f}%".format(np.mean(accuracies)*100))

    # Entrenar el modelo SVM con todos los datos de entrenamiento y evaluar en las imágenes de prueba
    svm.fit(train_features, train_labels)
    test_pred = svm.predict(test_features)

    for i, img in enumerate(test_imgs):
        # Obtener la etiqueta predicha para la imagen
        label = test_pred[i]
        # Asignar un texto a la imagen con la etiqueta predicha
        if label == 0:
            text = "Genuine"
        if label == 1:
            text = "Tampered"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Mostrar la imagen
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        








