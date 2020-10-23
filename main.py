import numpy as np
import cv2
import os

# Подготовка картинок
DIRPATH = 'train/'

files = os.listdir(DIRPATH)
images = []

for file_name in files:
    color_image = cv2.imread(DIRPATH + file_name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
    images = np.append(images, n_image)

images = images.reshape(14, 100)
print(images)

# Подготовка входныех параметров
training_set_inputs = images
training_set_outputs = np.array([[0],
                                 [1],
                                 [0],
                                 [1],
                                 [0],
                                 [1],
                                 [0],
                                 [1],
                                 [0],
                                 [1],
                                 [0],
                                 [1],
                                 [0],
                                 [1]])


# Нейросеть
class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((100, 1)) - 1

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)
            error = training_set_outputs - output
            adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            self.synaptic_weights = self.synaptic_weights + adjustment

    def think(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))


neural_network = NeuralNetwork()
print("Начальные случайные веса: ")
print(neural_network.synaptic_weights)

neural_network.train(training_set_inputs, training_set_outputs, 10000)

print("Веса после тренировки сети: ")
print(neural_network.synaptic_weights)

# Проверка результатов
NEW_DIRPATH = 'test/'

test_files = os.listdir(NEW_DIRPATH)
test_images = []

for file_name in test_files:
    color_image = cv2.imread(NEW_DIRPATH + file_name)
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
    test_images = np.append(test_images, n_image)

test_images = test_images.reshape(4, 100)
print("New situation:")
print(test_images)
print("Output data: ")
print(neural_network.think(test_images)[0])
print('[%.8f]' % neural_network.think(test_images)[1])
print(neural_network.think(test_images)[2])
print(neural_network.think(test_images)[3])
