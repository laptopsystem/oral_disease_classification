from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('oral_disease_classifier.h5')


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    img = request.files['image']
    img = image.load_img(img, target_size=(224, 224), color_mode='grayscale')
    img = image.img_to_array(img) / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred = model.predict(img)

    # Get the predicted class
    predicted_class = np.argmax(pred, axis=1)

    # Map class index to class name (assuming you have 4 classes)
    class_names = ['Healthy', 'Dental Caries', 'Gingivitis', 'Other']
    predicted_class_name = class_names[predicted_class[0]]

    return jsonify({'predicted_class': predicted_class_name})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
