import numpy as np
import os
import cv2
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import random
import pickle

DATADIR = "../data"
IMG_SIZE = 250
CATEGORIES = ["Viski", "Jager", "Stock"]

training_data = []
X = []
y = []

#čitanje i prilagodba slika
for category in CATEGORIES:
    path = os.path.join(DATADIR,category)
    class_name = category
    for img in tqdm(os.listdir(path)):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        training_data.append([new_array, class_name])

random.shuffle(training_data)

for features,label in training_data:
    X.append(features)
    y.append(label)

X= np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#pretvaranje label-a u one-hot array
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

#spremanje podataka u pickle datoteke
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()