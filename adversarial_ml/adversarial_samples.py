"""
author: Damiano Pasquini
email: damiano23@ru.is
course: Machine Learning in Cybersecurity @ RU
"""

import numpy as np
# import tensorflow as tf
from foolbox.attacks import SaliencyMapAttack
from foolbox.criteria import TargetClass
from foolbox.models import TensorFlowEagerModel
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras


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


def run_single_attack(kmodel, X_test, y_test, target):
    fmodel = TensorFlowEagerModel(kmodel, bounds=(0, 255))
    attack = SaliencyMapAttack(model=fmodel, criterion=TargetClass(target))
    image = np.array([X_test[0]])
    label = np.array([y_test[0]])
    adv_image = attack(image, label)
    # orig_predict = np.argmax(kmodel.predict(images))
    adv_predict = np.argmax(kmodel.predict(adv_image))
    return adv_image, adv_predict


def run_complete_attack(kmodel, X_test, y_test):
    fmodel = TensorFlowEagerModel(kmodel, bounds=(0, 255))
    attack = SaliencyMapAttack(model=fmodel, criterion=TargetClass(0))
    adv_samples = []
    for target_class in range(10):
        target_samples = []
        for i in range(len(X_test)):
            # TODO: check when to stop (when there are ten samples for each class)
            image = np.array([X_test[i]])
            label = np.array([y_test[i]])
            adv_image = attack(image, label)
            target_samples.append(adv_image)
            print(f"Target class: {target_class}, sample: {i}")
        adv_samples.append(target_samples)
    return adv_samples

def plot_matrix(adv):
    plt.subplot(10, len(adv), 1)
    plt.imshow(adv[0].reshape(28, 28), cmap='gray')
    plt.axis('off')
    plt.show()


X_train, X_test, y_train, y_test = pre_process()
model = load_model()
# adv_predict = run_single_attack(model, X_test, y_test, 1)
adv_matrix = run_complete_attack(model, X_test, y_test)
plot_matrix(adv_matrix)
np.save('adv_matrix.npy', adv_matrix)
