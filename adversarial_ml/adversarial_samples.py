"""
author: Damiano Pasquini
email: damiano23@ru.is
course: Machine Learning and Cybersecurity @ RU
"""

import numpy as np
import tensorflow as tf
from foolbox.attacks import SaliencyMapAttack
from foolbox.criteria import TargetClass
from foolbox.models import TensorFlowEagerModel
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import keras


def pre_process():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)  # reshape
    x_train = x_train.astype('float32') / 255  # normalize
    x_test = x_test.reshape(10000, 28, 28, 1)
    x_test = x_test.astype('float32') / 255
    y_train = to_categorical(y_train)  # one-hot encoding
    y_test = to_categorical(y_test)
    return x_train, x_test, y_train, y_test


def load_model(path="mnist_cnn_model.h5"):
    model = keras.models.load_model(path)
    return model


def run_attack(pretrained_model, x_test, y_test, target_class):
    fmodel = TensorFlowEagerModel(pretrained_model, bounds=(0, 255))  # instantiate a Foolbox model
    attack = SaliencyMapAttack(model=fmodel, criterion=TargetClass(target_class))  # Apply the SaliencyMapAttack
    input_samples = x_test[:10]
    original_labels = np.argmax(y_test[:10], axis=1)
    adversarial = attack(input_samples, original_labels)

    # Evaluate the model on the adversarial samples
    test_loss, test_acc = pretrained_model.evaluate(adversarial, y_test[:10])
    print(f'Test accuracy: {test_acc * 100:.2f}%')
    return adversarial


def plot_adversarial_samples(y_test, samples):
    adversarial = samples
    for i in range(10):
        plt.subplot(1, len(adversarial), i + 1)
        plt.imshow(adversarial[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {np.argmax(y_test[i])}')
        # todo print input label and adversarial label
        plt.axis('off')
    plt.show()

x_train, x_test, y_train, y_test = pre_process()
model = load_model()
attack_results = run_attack(model, x_test, y_test, 5)
plot_adversarial_samples(y_test, attack_results)
np.save('samples.npy', attack_results)
