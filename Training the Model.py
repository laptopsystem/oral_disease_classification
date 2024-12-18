from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Setup ImageDataGenerator for training and validation data
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Assuming you have a directory with images organized into subdirectories per class
train_data = train_datagen.flow_from_directory('path_to_train_data', target_size=(224, 224), color_mode='grayscale', batch_size=32, class_mode='categorical')

# For validation data
val_datagen = ImageDataGenerator(rescale=1./255)

val_data = val_datagen.flow_from_directory('path_to_val_data', target_size=(224, 224), color_mode='grayscale', batch_size=32, class_mode='categorical')

# Early stopping to avoid overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[early_stop])
