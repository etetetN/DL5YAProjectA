import array
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DL3 import *
import os
import cv2
import random
import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox

label_to_state = ["green", "red", "yellow"]

model = DLModel()
model.load_model("modelDir")
model.compile("categorical_cross_entropy")

def upload_image(img_label):
    file_path = filedialog.askopenfilename()
    if file_path:
        img_label.config(text=file_path)

class ModelUI(QtWidgets.QMainWindow):
    def upload_image(self):
        file_dialog = QFileDialog()
        file_dialog.setWindowTitle('Upload Image')
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg)')
        file_path = file_dialog.getOpenFileName(self, 'Upload Image', '', 'Images (*.png *.jpg *.jpeg)')[0]
        if file_path:
            pixmap = QtGui.QPixmap(file_path).scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.image_label.setPixmap(pixmap)
            self.input_path = file_path
        else:
            self.show_popup("Invalid file path! Make sure the path is valid and the image is valid.")

    def show_popup(self, message):
        popup = QMessageBox(self)
        popup.setWindowTitle('Popup Message')
        popup.setText(message)
        popup.exec_()

    def run_model(self):
        img_path = self.input_path
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if isinstance(img, (np.ndarray)): #Check if it loaded the image at all
            img1 = cv2.resize(img, (32, 52))
            new = []
            new.append(np.array(img1)) #The reason I don't directly use img1 is to fit the shape made for the model
            img_array = np.array(new)

            height, width, channels = img_array[0].shape

            img_array_1d = img_array.flatten().reshape(height*width*channels, 1)
            img_array_1d = np.array(img_array_1d) / 255.0 - 0.5

            pred = model.predict(img_array_1d)
            class_num = np.argmax(pred) #Sees where the prediction is 1 in the prediction array
            class_name = label_to_state[class_num]

            self.output_label.setText(f'This image has been predicted to be a {class_name} traffic light')
        else:
            self.show_popup("Invalid Image!")

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model UI')
        self.setGeometry(100, 100, 1000, 1000)

        self.image_label = QtWidgets.QLabel('No image uploaded')
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.image_label.setFrameStyle(QtWidgets.QFrame.Panel | QtWidgets.QFrame.Sunken)
        self.image_label.setLineWidth(1)
        self.image_label.setFixedSize(400, 400)

        self.run_button = QtWidgets.QPushButton('Run Model')
        self.run_button.clicked.connect(self.run_model)

        self.output_label = QtWidgets.QLabel('Model Output: ')
        self.output_label.setAlignment(QtCore.Qt.AlignCenter)

        self.upload_button = QtWidgets.QPushButton('Upload Image')
        self.upload_button.clicked.connect(self.upload_image)

        self.input_path = ""

        central_widget = QtWidgets.QWidget()
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.image_label, 0, 0, 1, 2)
        layout.addWidget(self.upload_button, 1, 0, 1, 1)
        layout.addWidget(self.run_button, 1, 1, 1, 1)
        layout.addWidget(self.output_label, 0, 2, 1, 1)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ModelUI()
    window.show()
    sys.exit(app.exec_())