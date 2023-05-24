X= []
y=[]

print("Cargando datos de imágenes falsificadas...")
for file_path in glob.glob(fake_images_path):
    img_name = file_path.split('\\')[-1]
    if  img_name.startswith('Tp_S_'):
        X.append(preparete_image_gaussians(file_path, image_size))
        y.append(0)
    else:
        X.append(preparete_image_gaussians(file_path, image_size))
        y.append(1)

print("X shape: ", np.array(X).shape)
print("y shape: ", np.array(y).shape)

X= np.array(X).reshape(-1, image_size[0], image_size[1], 3)
y = to_categorical(y, num_classes = 2)


'''X_Tp,Y_Tp= create_dataset_for_tampered_images('../data/dataset/data/CASIA2/Tp', image_size)

print("X shape: ", np.array(X_Tp).shape)
print("y shape: ", np.array(Y_Tp).shape)

X_Tp= np.array(X_Tp).reshape(-1, image_size[0], image_size[1], 3)
Y_Tp = to_categorical(Y_Tp, num_classes = 2)'''

# Entrenamos con 80% de los datos de train, usar 10% para validación y 10% para test
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size = 0.2, random_state=5)
X_val, X_test, Y_val, Y_test = train_test_split(X_val, Y_val, test_size = 0.5, random_state=5)

model = build_splice_classification_model_v2(X_train, Y_train, X_val, Y_val,image_size)

# Calculamos la precisión del modelo con los datos de test
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])