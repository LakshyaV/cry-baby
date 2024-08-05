import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps

model = tf.keras.models.load_model('./modelTwo.h5')

def image_gen(audio_path, save_path):
    y, sr = librosa.load(audio_path, sr=None)

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel-Spec of {os.path.basename(audio_path)}")
    plt.tight_layout()
    save_path = os.path.join(save_path, f"{os.path.basename(audio_path)}.png")
    plt.savefig(save_path)
    plt.close()

    return save_path

def preprocess_image(img_path):
    target_height, target_width = 256, 128  # Ensure these match the model's expected input size
    img = image.load_img(img_path, target_size=(target_height, target_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def get_most_recent_file(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    most_recent_file = max(files, key=os.path.getctime)
    return most_recent_file

'''
directory = './audio_files/'
audio_path = get_most_recent_file(directory)

img_array = image_gen(audio_path, "./mel_specs/")
img_array = preprocess_image(img_array)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Map predicted class index to class label
class_labels = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
predicted_label = class_labels[predicted_class]
print(f'Predicted class: {predicted_label}')
'''