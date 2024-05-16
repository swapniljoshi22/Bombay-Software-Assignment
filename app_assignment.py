from flask import Flask, render_template, request
import cv2
import numpy as np
import onnxruntime

app = Flask(__name__)

# Load the ONNX model
ort_session = onnxruntime.InferenceSession("svm_model.onnx")

# Function to process the uploaded image and make predictions
def process_image(image):
    # Preprocess the image (resize, convert to grayscale, flatten)
    img = cv2.resize(image, (64, 64))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_vector = img_gray.flatten()
    # Convert input data to ONNX-compatible format
    input_data = {"input": img_vector.astype(np.float32)}
    # Run inference
    pred = ort_session.run(None, input_data)
    # Extract predicted label
    label = np.argmax(pred)
    return label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        file = request.files['file']
        if file:
            # Read the uploaded image
            img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            # Process the image and get the prediction
            label = process_image(img)
            return f"Predicted label: {label}"
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
