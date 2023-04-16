import numpy as np
np.random.seed(2)
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


def build_model(X_train, Y_train, X_val, Y_val):
    model = Sequential()
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 3)))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()

    epochs = 40
    batch_size = 32
    init_lr = 1e-4
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 2,verbose = 0,mode = 'auto')
    
    model.fit(X_train,Y_train,batch_size = batch_size, epochs = epochs,validation_data = (X_val, Y_val), callbacks = [early_stopping])
    
    #calculamos la precisión del modelo
    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('models/model'+str(score[1])+'.h5')

    return model


def build_splice_classification_model(X_train, Y_train, X_val, Y_val):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'valid', activation = 'relu', input_shape = (128, 128, 1)))
    model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'valid', activation = 'relu'))
    model.add(MaxPool2D(pool_size = (2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'softmax'))

    model.summary()

    epochs = 40
    batch_size = 32
    init_lr = 1e-4
    opt = Adam(lr=init_lr, decay=init_lr / epochs)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 2,verbose = 0,mode = 'auto')
    
    model.fit(X_train,Y_train,batch_size = batch_size, epochs = epochs,validation_data = (X_val, Y_val), callbacks = [early_stopping])

    #calculamos la precisión del modelo
    score = model.evaluate(X_val, Y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.save('models/splice_model'+str(score[1])+'.h5')

    return model