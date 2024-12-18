# Evaluate the model on the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory('path_to_test_data', target_size=(224, 224), color_mode='grayscale', batch_size=32, class_mode='categorical')

test_loss, test_acc = model.evaluate(test_data)
print(f'Test Accuracy: {test_acc}')
