
import cv2
import tensorflow as tf

DATADIR = "../ruap_data/test/"
IMG_SIZE = 350
IMG_NAME = input("Unesite ime datoteke: ")
#prilagodba slike za testiranje
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#učitavanje modela
model = tf.keras.models.load_model("64x3-CNN.model")

#poziv prilagodbi slike
prediction = model.predict([prepare(DATADIR + IMG_NAME)])

print(prediction)
#ispis predviđanja - u nekim slučajevima se vraća niz u kojem je max vrijednost 0.9999....
#te se zbog toga koristi funkcija max() da bismo dobili ispis
if prediction[0][0] == max(prediction[0]):
    print("Piće na slici je Jager")
elif prediction[0][1] == max(prediction[0]):
    print("Piće na slici je Stock")
elif prediction[0][2] == max(prediction[0]):
    print("Piće na slici je viski")

