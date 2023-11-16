'''
a program to classify jellyfish species using a custom convolutional neural network
'''

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras_tuner import HyperParameters

# module imports
from load_preprocess import format_data, generate_aug_data
from visuals import vis_imgs, vis_imgs_dist, vis_metrics, vis_model_cm
from models import lr_schedule, create_cnn, model_tuner, opt_batch_size

### load in data 

types = ['barrel_jelly', 'blue_jelly', 'compass_jelly', 'mane_jelly', 'stinger_jelly', 'moon_jelly']

# load in images as numpy arrays
imgs_count, X, y = format_data(types)

# Normalize pixel values
X_norm = X.astype('float32') / 255.0

# split the data
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.25, random_state=34)

vis_imgs(X_train, y_train, 'originals')

# observe distribution of image classes
train_cats = np.unique(y_train, return_counts=True)[1]
test_cats = np.unique(y_test, return_counts=True)[1]

vis_imgs_dist(train_cats, test_cats)

# augment images to expand training dataset
X_train_combined, y_train_combined = generate_aug_data(X_train, y_train, 200)

# observe distribution of image classes with augmentations
train_cats_aug = np.unique(y_train_combined, return_counts=True)[1]
test_cats_aug = np.unique(y_test, return_counts=True)[1]

vis_imgs_dist(train_cats_aug, test_cats_aug, 'after augmentation')

# one-hot encode labels for model
y_train_combined = to_categorical(y_train_combined, num_classes=6)
y_test = to_categorical(y_test, num_classes=6)

### create baseline CNN model

largest_class = np.argmax(train_cats)
num_classes = len(train_cats)
print(f'baseline: {largest_class / num_classes}')

# define early stopping callback
early_stopping = EarlyStopping(monitor='val_accuracy',
                               patience=5,
                               restore_best_weights=True)

# define learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

callbacks = [early_stopping, lr_scheduler]

# instantiate cnn model
base_model = create_cnn()
print(base_model.summary())

# fit model with early stopping
baseline = base_model.fit(X_train_combined,
                          y_train_combined,
                          epochs=999,
                          validation_split=0.25,
                          shuffle=True,
                          workers=4,  # set for Apple M2 chip
                          callbacks=callbacks)

vis_metrics(baseline)
vis_model_cm(base_model, X_test, y_test)

### build out hyperparameter dictionary

'''
hp = HyperParameters()
hp.Choice('optimizer', values=['adam', 'sgd'])
hp.Int('filters', min_value=24, max_value=48,step=4)
hp.Choice('kernel_size', values=[3, 5, 7])
hp.Choice('activation',values=['relu', 'sigmoid'])
hp.Fixed('kernel_init', 'he_uniform')
hp.Float('dropout', min_value=0.2, max_value=0.6, step=0.1)
hp.Choice('decay', values=[1e-4, 1e-3])

# find best parameters
# this can take several days

results = model_tuner(create_cnn,
                      X_train_combined,
                      y_train_combined,
                      hp)

best_hp = results.get_best_hyperparameters()[0]
print(best_hp)
'''

# resolved best hparams
best_hp = HyperParameters()
best_hp.Fixed('optimizer', 'sgd')
best_hp.Fixed('filters', 44)
best_hp.Fixed('kernel_size', 5)
best_hp.Fixed('activation', 'relu')
best_hp.Fixed('kernel_init', 'he_uniform')
best_hp.Fixed('dropout', 0.4)
best_hp.Fixed('decay', 1e-4)

# determine opt batch size

best_model = create_cnn(hp=best_hp)
print(best_model.summary())

'''
size_range = [i for i in range(24, 48, 4)]
opt_size = opt_batch_size(best_model,
                          X_train_combined,
                          y_train_combined,
                          size_range)
'''

# resolved best batch size
opt_size = 36

### implement best HPs in final model

best_hist = best_model.fit(X_train_combined,
                           y_train_combined,
                           batch_size=opt_size,
                           epochs=999,
                           validation_split=0.25,
                           shuffle=True,
                           workers=4,  # set for Apple M2 chip
                           callbacks=callbacks)

vis_metrics(best_hist)
vis_model_cm(best_model, X_test, y_test)
