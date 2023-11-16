'''
a module containing helper functions for visualizing model performance
'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# extract specific species from label
types = ['barrel_jelly', 'blue_jelly', 'compass_jelly', 'mane_jelly', 'stinger_jelly', 'moon_jelly']
types = [type_[:type_.find('_')] for type_ in types]

def vis_imgs(image_list, labels=None, tag='original'):
    '''
    display function to visualize the first five images from a list

    parameters:
        image_list: (list) of images
        labels: (list) of labels

    returns:
        None (displays the images)
    '''

    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(image_list[i])
        plt.title(types[labels[i]])
        plt.axis('off')

    plt.suptitle(f'{tag} images sample')
    plt.show()


def vis_imgs_dist(y_train, y_test, tag='Baseline'):
    '''
    display function to visualize distribution of images in train and test set for each class

    parameters:
        X_train: (numpy array) of images in training set
        X_test: (numpy array) of images in test set

    returns:
        None (displays stacked bar chart)
    '''

    plt.bar(types, y_train, color='y')
    plt.bar(types, y_test, bottom=y_train, color='c')

    plt.title(f'{tag} number of observations for each species of Jellyfish')
    plt.xlabel('Jellyfish species')
    plt.ylabel('Count')
    plt.legend(['training set', 'test set'])

    plt.show()


def vis_metrics(model):
    '''
    display function to visualize neural network metrics

    parameters:
        model: (keras history object)

    returns:
        None (displays plot)
    '''

    metrics = {'loss': model.history['loss'],
               'accuracy': model.history['accuracy'],
               'val_loss': model.history['val_loss'],
               'val_accuracy': model.history['val_accuracy']}
    
    n_epochs = len(metrics.get('accuracy'))

    epochs = [i for i in range(n_epochs)]
    sns.lineplot(x=epochs, y=metrics.get('accuracy'), label='training set')
    sns.lineplot(x=epochs, y=metrics.get('val_accuracy'), label='validation set')
    plt.title('Change in prediction accuracy thru epochs')
    plt.grid()

    plt.show()

def vis_model_cm(model, X_test, y_test):
    '''
    display function to visualize confusion matrix of model predictions

    parameters:
        model: Trained machine learning model
        X_test: Test data
        y_test: True labels for the test data
        class_names: List of class names

    returns:
        None (displays the confusion matrix)
    '''

    # Get model predictions on the test data
    y_pred = model.predict(X_test)

    # Convert one-hot encoded labels back to categorical labels
    y_test_labels = [types[i] for i in np.argmax(y_test, axis=1)]
    y_pred_labels = [types[i] for i in np.argmax(y_pred, axis=1)]

    # Generate confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=types)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(len(types), len(types)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=types, yticklabels=types)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
