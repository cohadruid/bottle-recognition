import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

CATEGORIES = ["Viski", "Jager", "Stock"]  # will use this to convert prediction num to string value
DATADIR = "data/train"
IMG_SIZE = 250
IMG_NAME = input("Unesite ime datoteke: ")
encoder = LabelEncoder()

#prilagodba slike za testiranje
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#učitavanje modela
model = tf.keras.models.load_model("64x6-CNN.model")

#poziv prilagodbi slike
prediction = model.predict([prepare('data/test/' + IMG_NAME)])

#ispis predviđanja
if prediction[0][0] == 1:
    print("Piće na slici je Jager")
elif prediction[0][1] == 1:
    print("Piće na slici je Stock")
elif prediction[0][2] == 1:
    print("Piće na slici je viski")
