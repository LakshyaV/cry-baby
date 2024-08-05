import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os

train_dir = "./split_data/train/"
test_dir = "./split_data/test/"

img_height, img_width = 256, 128
batch_size = 32
classes = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

val_generator = val_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='sparse'
)

'''
class_counts = []
for i in os.listdir("./resized_mel_specs/"):
    counter = 0
    new_dir = "./resized_mel_specs/" + i
    for j in os.listdir(new_dir):
        counter += 1
    class_counts.append(counter)
'''

#new_count = [i for i in range(len(class_counts)) for j in range(class_counts[i])]
class_counts = [48, 48, 54, 382, 48]
new_count = [i for i in range(len(class_counts)) for j in range(class_counts[i])]

class_labels = np.arange(len(class_counts))
class_weights = compute_class_weight('balanced', classes=class_labels, y=new_count)
class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the learning rate scheduler
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 0.95 ** epoch)

# Define early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weight_dict
)

loss, accuracy = model.evaluate(val_generator)
print(f"Validation Accuracy: {accuracy:.4f}")
# model.save('./modelSeven.h5')