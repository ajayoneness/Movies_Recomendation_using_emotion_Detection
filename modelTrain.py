import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Define hyperparameters
batch_size = 32
epochs = 50
input_shape = (48, 48, 1)  # Assumes grayscale images of size 48x48

# Define model architecture
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))  # Assumes 7 emotion classes

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define data generators for training and validation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('path_to_training_data_directory', target_size=input_shape[:2],
                                                    color_mode='grayscale', batch_size=batch_size,
                                                    class_mode='categorical')
val_generator = val_datagen.flow_from_directory('path_to_validation_data_directory', target_size=input_shape[:2],
                                                color_mode='grayscale', batch_size=batch_size,
                                                class_mode='categorical')

# Define model checkpoint for saving best model during training
checkpoint = ModelCheckpoint('path_to_your_emotion_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# Train the model
model.fit(train_generator, steps_per_epoch=train_generator.samples // batch_size, epochs=epochs,
          validation_data=val_generator, validation_steps=val_generator.samples // batch_size,
          callbacks=[checkpoint])

# Save the final trained model
model.save('path_to_your_emotion_model.h5')
