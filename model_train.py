
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DL3 import *
import os
import cv2
import random

label_to_state = ["green", "red", "yellow"]

def load_images_from_folder(folder):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        if subfolder == 'green':
            label = 0
        elif subfolder == 'red':
            label = 1
        else:
            label = 2

        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img = cv2.imread(os.path.join(subfolder_path, filename), cv2.IMREAD_COLOR)
            if img.any():
                img1 = cv2.resize(img, (32, 52))
                images.append(np.array(img1))
                curr_label_array = np.zeros(3)
                curr_label_array[label] = 1
                labels.append(curr_label_array)
    return np.array(images), np.array(labels).T

def show_random_example_images(images, labels, num_examples=5):
    num_images = len(images)
    random_indices = np.random.choice(num_images, size=num_examples, replace=False)
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    for i, idx in enumerate(random_indices):
        axes[i].imshow(images[idx])
        #axes[i].set_title(label_to_state[labels[idx]])
        axes[i].axis('off')
    plt.show()


training_images, training_labels = load_images_from_folder('traffic_light_data/train')
testing_images, testing_labels = load_images_from_folder('traffic_light_data/val')

# training examples
show_random_example_images(training_images, training_labels)

# testing examples
show_random_example_images(testing_images, testing_labels)

print("-------------------------------------")

training_images_num = training_images.shape[0]
testing_images_num = testing_images.shape[0]

print(f"training images amount: {training_images_num}")
print(f"training images amount: {testing_images_num}")
print(f"an image shape: {training_images[0].shape}")

print("-------------------------------------")

height, width, channels = training_images[0].shape #All images have the same height, width, amount of channels (rgb)

training_images_1d = training_images.flatten().reshape(training_images_num, width*height*channels).T
testing_images_1d = testing_images.flatten().reshape(testing_images_num, width*height*channels).T

training_images_1d = np.array(training_images_1d) / 255.0 - 0.5
testing_images_1d = np.array(testing_images_1d) / 255.0 - 0.5

print(f"training images new shape: {training_images_1d.shape}")
print(f"training images new shape: {testing_images_1d.shape}")

print("-------------------------------------")
np.random.seed(42)

hidden1 = DLLayer("Perseptrons 1", 32,(training_images_1d.T.shape[1], ),"trim_softmax",W_initialization = "Xaviar",learning_rate = 0.01, random_scale=0.015)
hidden2 = DLLayer ("Perseptrons 2", 16,(32,),"trim_tanh",W_initialization = "Xaviar",learning_rate = 0.1, optimization="adaptive")
#hidden3 = DLLayer("Perspetrons 3", 8, (10,), "trim_sigmoid", W_initialization="He", learning_rate=0.1, optimization="adaptive")
#hidden4 = DLLayer("Perspetrons 4", 4, (8,), "relu", W_initialization="He", learning_rate=0.075, random_scale=0.005)
output = DLLayer("Output", 3, (16, ), "trim_softmax", W_initialization="He", learning_rate=0.1)

model = DLModel()
model.add(hidden1)
model.add(hidden2)
#model.add(hidden3)
#model.add(hidden4)
model.add(output)

print(model)

print("-------------------------------------")

model.compile("categorical_cross_entropy")

costs = model.train(training_images_1d, training_labels, 850)

print("-------------------------------------")

plt.plot(costs)
plt.ylabel('Costs')
plt.xlabel('Iterations %')
plt.title('Cost Over Time')
plt.show() #Graph for costs over time

print("-------------------------------------")

model.confusion_matrix(testing_images_1d, testing_labels)
model.save_weights('modelDir')

print("-------------------------------------")
predictions_test = model.predict(testing_images_1d)
predictions_train = model.predict(training_images_1d)

training_accuracy = np.sum(np.argmax(predictions_train, axis=0) == np.argmax(training_labels, axis=0)) / training_labels.shape[1] #For green light accuracy should be 100% for safety lol
testing_accuracy = np.sum(np.argmax(predictions_test, axis=0) == np.argmax(testing_labels, axis=0)) / testing_labels.shape[1] #For green light accuracy should be 100% for safety lol

print(f"Test Accuracy: {testing_accuracy}")
print(f"Train Accuracy: {training_accuracy}")

