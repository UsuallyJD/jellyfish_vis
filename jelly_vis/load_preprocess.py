'''
a module containing helper functions for loading and preprocessing images
'''

import os
import cv2
import numpy as np

from keras.preprocessing.image import ImageDataGenerator

# module imports
from visuals import vis_imgs

# Function to load and preprocess images
def load_img(folder):
    '''
    loading function to get images from directories

    parameters:
        folder: (string) path to directory

    returns:
        img_list: (list) of images
    '''
    img_list = []
    for filename in os.listdir(folder):

        img = cv2.imread(os.path.join(folder, filename))

        if img is not None:
            # resize to a fixed size for the model
            img = cv2.resize(img, (224, 224))
            img_list.append(img)

    return img_list


def format_data(classes):
    '''
    reformatting function to cast images to model-compatible format

    parameters:
        classes: (list) of class name strings (which are also image directory names)

    returns:
        img_count: (list) with number of images for each class
        img_list: (list) of all images in class order
        img_labels: (list) of class labels for each image
    '''

    i = 0
    img_list = []
    img_labels = []
    img_count = []

    for _class in classes:
        imgs = load_img(f'data/{_class}')

        img_count.append(len(imgs))
        img_list += imgs

        labels = [i] * len(imgs)
        img_labels += labels

        i += 1
    
    return img_count, np.array(img_list), np.array(img_labels)


def generate_aug_data(X_train, y_train, n_steps=100):
    '''
    generating function to create augmented images

    parameters:
        X_train: (np.array) of training images
        y_train: (np.array) of training labels
        n_steps: (int) number of augmented images to generate per image
    
    returns:
        X_train_combined: (np.array) of combined training images
        y_train_combined: (np.array) of combined training labels
    '''

    # create an ImageDataGenerator instance
    datagen = ImageDataGenerator(rotation_range=40,
                                width_shift_range=0.25,
                                height_shift_range=0.25,
                                shear_range=0.25,
                                zoom_range=0.25,
                                horizontal_flip=True)
 
    # generate augmented images
    augmented_data = datagen.flow(X_train, y_train, batch_size=32)

    # initialize lists to store augmented data
    augmented_images = []
    augmented_labels = []

    # generate augmented data
    for _ in range(n_steps):
        batch = augmented_data.next()  # get next batch
        augmented_images.extend(batch[0])  # all images
        augmented_labels.extend(batch[1])  # all labels

    # visualize augmented images
    vis_imgs(augmented_images, augmented_labels, 'augmented')

    # combine data
    X_train_combined = np.vstack((X_train, augmented_images))
    y_train_combined = np.append(y_train, augmented_labels)

    return X_train_combined, y_train_combined
