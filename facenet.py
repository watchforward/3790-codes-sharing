import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionResNetV2

# path of data set
dataset_dir = 'E:/3790testenvironment/python/face/2/archive'

# size of images
img_height, img_width = 160, 160

# after process storing
processed_dir = 'E:/3790testenvironment/python/face/facenet_processed'
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)


for person in os.listdir(dataset_dir):
    person_dir = os.path.join(dataset_dir, person)
    if os.path.isdir(person_dir):
        processed_person_dir = os.path.join(processed_dir, person)
        if not os.path.exists(processed_person_dir):
            os.makedirs(processed_person_dir)
        for img_name in os.listdir(person_dir):
            if img_name.endswith('.pgm'):
                img_path = os.path.join(person_dir, img_name)
                img = Image.open(img_path)
                img = img.resize((img_width, img_height))
                img = img.convert('RGB')
                img.save(os.path.join(processed_person_dir, img_name.replace('.pgm', '.jpg')))

print('preparing completed')

# generator
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    processed_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    processed_dir,
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# lad the model
base_model = InceptionResNetV2(include_top=False, input_shape=(img_height, img_width, 3))

# use the dense to link
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

# construct the model
model = Model(inputs=base_model.input, outputs=output)


for layer in base_model.layers:
    layer.trainable = False

# compiling
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# training
epochs = 10

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# drawing
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# test
val_loss, val_acc = model.evaluate(validation_generator)
print(f'Validation Accuracy: {val_acc}')

# save
model.save('facenet_recognition_model.h5')
