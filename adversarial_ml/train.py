"""
author: Damiano Pasquini
email: damiano23@ru.is
course: Machine Learning and Cybersecurity @ RU
"""

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# data preprocessing
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN model
model = models.Sequential() # Sequential model
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # Convolutional layer
model.add(layers.MaxPooling2D((2, 2))) # Max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Convolutional layer
model.add(layers.MaxPooling2D((2, 2))) # Max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # Convolutional layer
model.add(layers.Flatten()) # Flatten layer
model.add(layers.Dense(64, activation='relu')) # Dense layer
model.add(layers.Dense(10, activation='softmax')) # Dense layer

# model compiled
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# training the model
model.fit(x_train, y_train, epochs=5, batch_size=64)

# model evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc * 100:.2f}%')

# checks if the model meets the 98% accuracy threshold
if test_acc >= 0.98:
    model.save('mnist_cnn_model.h5')
    print("Model saved to 'mnist_cnn_model.h5'")
else:
    print("Model did not meet the accuracy threshold.")
