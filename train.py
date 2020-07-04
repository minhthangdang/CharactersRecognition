from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
import os
from model import model

# Define constants
DATASET_PATH = './English/Fnt/'
MODEL_PATH = '.'
BATCH_SIZE = 128
EPOCHS = 20
TARGET_WIDTH = 128
TARGET_HEIGHT = 128
TARGET_DEPTH = 3

# Set up the data generator to flow data from disk
print("[INFO] Setting up Data Generator...")
data_gen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = data_gen.flow_from_directory(
    DATASET_PATH, 
    subset='training',
    target_size = (TARGET_WIDTH, TARGET_HEIGHT),
    batch_size = BATCH_SIZE
)

val_generator = data_gen.flow_from_directory(
    DATASET_PATH,
    subset='validation',
    target_size = (TARGET_WIDTH, TARGET_HEIGHT),
    batch_size = BATCH_SIZE
)

# Build model
print("[INFO] Compiling model...")
alexnet = model(train_generator.num_classes, (TARGET_WIDTH, TARGET_HEIGHT, TARGET_DEPTH))

# Compile the model
alexnet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the network
print("[INFO] Training network ...")
# Set the learning rate decay
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, min_lr=0.001)
H = alexnet.fit_generator(
	train_generator,
	validation_data=val_generator,
	steps_per_epoch=train_generator.samples // BATCH_SIZE,
	validation_steps = val_generator.samples // BATCH_SIZE,
	epochs=EPOCHS, verbose=1, callbacks=[reduce_lr])

# save the model to disk
print("[INFO] Serializing network...")
alexnet.save(MODEL_PATH + os.path.sep + "trained_model")

print("[INFO] Done!")