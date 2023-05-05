from flask import Flask, request, jsonify
import onnxruntime
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__, template_folder='templates')
# app = Flask(__name__)
session = onnxruntime.InferenceSession('model.onnx')

@app.route('/')
def home():
    return render_template('index.html', pred="Login", vis ="visible")

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    elif request.method == 'POST':
        # Get the image file from the request
        file = request.files.get('file')
        if file is None:
            return jsonify({'error': 'No file in request'}), 400

        # Load the image using PIL
        img = Image.open(file).convert('RGB')
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2)).astype('float32')

        # Run the model on the image
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        output = session.run([output_name], {input_name: img})[0]

        # Convert the output to a human-readable label
        label = np.argmax(output)

        # Return the label as a JSON response
        return jsonify({'prediction': str(label)})


if __name__ == '__main__':
    app.run(debug=True)


