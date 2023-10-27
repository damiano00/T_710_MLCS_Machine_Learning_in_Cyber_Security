"""
author: Damiano Pasquini
email: damiano23@ru.is
course: Machine Learning in Cybersecurity @ RU
"""

import numpy as np
import matplotlib.pyplot as plt
import keras
from foolbox.attacks import SaliencyMapAttack
from foolbox.criteria import TargetClass
from foolbox.models import TensorFlowEagerModel
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical


def pre_process():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)  # reshape
    X_train = X_train.astype('float32') / 255  # normalize
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train, 10)  # one-hot encoding
    y_test = to_categorical(y_test, 10)
    return X_train, X_test, y_train, y_test


def load_model(path="mnist_cnn_model.h5"):
    model = keras.models.load_model(path)
    return model


def run_single_attack(kmodel, X_test, y_test, target, index=0):
    fmodel = TensorFlowEagerModel(kmodel, bounds=(0, 255))
    attack = SaliencyMapAttack(model=fmodel, criterion=TargetClass(target))
    image = np.array([X_test[index]])
    label = np.array([y_test[index]])
    adv_image = attack(image, label)
    adv_predict = np.argmax(kmodel.predict(adv_image))
    return image, label, adv_image, adv_predict


def run_complete_attack(kmodel, X_test, y_test):
    mat = [[None for _ in range(10)] for _ in range(10)]
    for i in range(len(X_test)): # for each image in the test set
        if not all(all(element is not None for element in row) for row in mat):  # if mat is not complete
            for target in range(10): # for each target
                _, label, adv_image, pred = run_single_attack(kmodel, X_test, y_test, target, i)
                if all(element is not None for element in mat[np.argmax(label)]):  # if row is complete
                    print("This target ("+str(np.argmax(label))+") is complete")
                    break
                if pred != target:  # if attack failed
                    print("No adversarial image found for target:", target)
                    mat[np.argmax(label)] = [None] * 10
                    break
                if target == np.argmax(label):  # if target is the same as the label
                    mat[np.argmax(label)][target] = X_test[i]
                else:
                    mat[np.argmax(label)][target] = adv_image
        else:
            return mat


def plot_number(image):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.show()


def plot_matrix(matrix):
    fig, axs = plt.subplots(10, 10)
    for i in range(10):
        for j in range(10):
            axs[i, j].imshow(matrix[i][j].reshape(28, 28), cmap='gray')
            axs[i, j].axis('off')
    plt.tight_layout()
    plt.savefig("adversarial_samples.png")
    plt.show()


X_train, X_test, y_train, y_test = pre_process()
model = load_model()
mat = run_complete_attack(model, X_test, y_test)
plot_matrix(mat)
