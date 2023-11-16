'''
a module containing helper functions for creating computer vision models
'''

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout
# legacy optimizers for apple silicon
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.regularizers import l2 as weight_decay
from keras.callbacks import EarlyStopping, LearningRateScheduler

from keras_tuner import HyperParameters
from kerastuner.tuners import RandomSearch

# default hyperparameters for basic model
default = HyperParameters()
default.Fixed('learn_rate', 1e-2)
default.Fixed('optimizer', "adam")
default.Fixed('filters', 32)
default.Fixed('kernel_size', 3)
default.Fixed('activation', "sigmoid")
default.Fixed('kernel_init', "glorot_uniform")
default.Fixed('dropout', 0)
default.Fixed('decay', 0)


# learning rate schedule
def lr_schedule(epoch):
    '''
    calculator function to base learning rate on epoch

    parameters:
        epoch: (int) number of epochs
    
    returns:
        lr: (float) learning rate
    '''

    init_lr = 0.01
    drop = 0.5
    epochs_drop = 8.0
    
    lr = init_lr * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))

    return lr.numpy()


# early stopping
early_stopping = EarlyStopping(monitor='val_accuracy',
                                patience=3,
                                restore_best_weights=True)

# learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [early_stopping, lr_scheduler]


# create convolutional neural network model
def create_cnn(hp=default):
    '''
    generator function to create a convolutional neural network

    Parameters:
        hp: (keras object) hyperparameters for model
        lr: (float) Learing rate for Stocastic Gradient Descent

    Returns:
        model: (keras object) compiled model
    '''

    opt = hp.get('optimizer')
    if opt == "adam":
        optimizer = Adam(0.01)
    elif opt == "sgd":
        optimizer = SGD(0.01)
    else:
        print (f'{opt} is not a valid optimizer. Default to Adam')
        optimizer = Adam(0.01)

    model = Sequential([Conv2D(filters=hp.get('filters'),
                               kernel_size=hp.get('kernel_size'),
                               activation=hp.get('activation'),
                               kernel_initializer=hp.get('kernel_init'),
                               input_shape=(224, 224, 3),
                               # add weight decay for smoother regularization
                               kernel_regularizer=weight_decay(hp.get('decay'))),
                        BatchNormalization(),
                        MaxPooling2D((2, 2)),
                        # dropout 1/4 of nodes in above layer
                        Dropout(hp.get('dropout')),
                        Conv2D(filters=hp.get('filters'),
                               kernel_size=hp.get('kernel_size'),
                               activation=hp.get('activation'),
                               kernel_initializer=hp.get('kernel_init'),
                               input_shape=(224, 224, 3),
                               # add weight decay for smoother regularization
                               kernel_regularizer=weight_decay(hp.get('decay'))),
                        BatchNormalization(),
                        MaxPooling2D((2, 2)),
                        # dropout 1/4 of nodes in above layer
                        Dropout(hp.get('dropout')),
                        Flatten(),
                        Dense(6, activation='softmax')
])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics='accuracy')

    return model


def model_tuner(model_template, X_train, y_train, hp):
    '''
    function to tune hyperparameters of a model

    Parameters:
        None

    Returns:
        tuner results summary
    '''

    random_tuner = RandomSearch(model_template,
                                objective='val_accuracy',
                                max_trials=100,
                                seed=3204, # set for reproducibility (and signing my work)
                                hyperparameters=hp,
                                directory='random_search',
                                project_name='jellyfish')

    random_tuner.search(X_train,
                        y_train,
                        epochs=999,
                        validation_split=0.25,
                        callbacks=callbacks)

    return random_tuner.results_summary()


def opt_batch_size(model, X, y, batch_sizes):
    '''
    function to find optimal batch size for passed model

    parameters:
        model: (keras object) compiled model
        X_train: (numpy array) training data
        y_train: (numpy array) training labels
        batch_sizes: (list) of batch sizes to test
    
    returns:
        opt_size: (int) optimal batch size
    '''

    opt_size = None
    best_val_acc = 0.0

    # loop thru batch sizes testing model w/each
    for size in batch_sizes:

        # train model on batch_size
        history = model.fit(X,
                            y,
                            batch_size=size,
                            epochs=999,
                            validation_split=0.25,
                            shuffle=True,
                            workers=4,  # set for Apple M2 chip
                            callbacks=callbacks)

        # get best validation accuracy
        val_accuracy = max(history.history['val_accuracy'])

        # update overall best if current batch size is better
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            opt_size = size

    return opt_size
