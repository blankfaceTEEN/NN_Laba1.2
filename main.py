import numpy as np
import cv2
import os


# Подготовка картинок
def preprocess(path, count):
    files = os.listdir(path)
    buffer_images = []

    for file_name in files:
        color_image = cv2.imread(path + file_name)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        n_image = np.around(np.divide(gray_image, 255.0), decimals=1)
        buffer_images = np.append(buffer_images, n_image)

    buffer_images = buffer_images.reshape(count, 100)
    print(buffer_images)
    return buffer_images


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


images = preprocess('train/', 14)

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

neural_network = NeuralNetwork()
print("\nНачальные случайные веса: ")
print(neural_network.synaptic_weights)

neural_network.train(training_set_inputs, training_set_outputs, 10000)

print("\nВеса после тренировки сети: ")
print(neural_network.synaptic_weights)

# Проверка результатов
test_images = preprocess('test/', 4)

print("\nНовые картинки:")
print(test_images)
print("\nРезультат нейросети: ")
print('[%.8f]' % neural_network.think(test_images)[0])
print('[%.8f]' % neural_network.think(test_images)[1])
print('[%.8f]' % neural_network.think(test_images)[2])
print('[%.8f]' % neural_network.think(test_images)[3])
