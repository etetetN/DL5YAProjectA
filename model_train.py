
from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import h5py
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from DL3 import *



hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xaviar",learning_rate = 0.0075)
hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir","Hidden1")
hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization =
"SaveDir/Hidden1.h5",learning_rate = 0.1)
hidden3 = DLLayer("Perspetrons 3", 10, (10,), "softmax", W_initialization="Xaviar", learning_rate=0.15)
model = DLModel()
model.add(hidden1)
model.add(hidden2)
model.add(hidden3)
print(model)
dir = "mo32del"
model.save_weights(dir)
print(os.listdir(dir))