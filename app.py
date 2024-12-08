"""from flask import Flask, request, jsonify, render_template
from keras.models import load_model
#from keras.preprocessing import image
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('cats_dogs_model')

def process_image(input_image_path):
    img = Image.open(input_image_path)  # Resize the image to match model's expected sizing
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the pixel values

    # Make prediction using the loaded model
    prediction = model.predict(img_array)

    # Determine class label based on prediction
    class_label = "cat" if prediction < 0.5 else "dog"
    return class_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            img = image.load_img(file, target_size=(200, 200))
            class_label = process_image(img)
            return render_template('index.html', class_label=class_label, filename=file.filename)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
"""

"""from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('cats_dogs_model')

def process_image(input_image_path):
    # Load and preprocess the input image using Pillow (PIL)
    img = Image.open(input_image_path)
    img = img.resize((200, 200))  # Resize the image to match your model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the pixel values

    # Make prediction using the loaded model
    prediction = model.predict(img_array)

    # Determine class label based on prediction
    class_label = "cat" if prediction < 0.5 else "dog"

    # Output message based on the prediction
    if class_label == "cat":
        return "It's a cat!"
    else:
        return "It's a dog!"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            img_path = "F:/cat dog/test_set/test_set/cats/" + file.filename
            output_message = process_image(img_path)
            return render_template('index.html', class_label=output_message, filename=file.filename)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)"""
    
from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = load_model('cats_dogs_model')

def process_image(input_image_path):
    # Load and preprocess the input image using Pillow (PIL)
    img = Image.open(input_image_path)
    img = img.resize((200, 200))  # Resize the image to match your model's input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the pixel values

    # Make prediction using the loaded model
    prediction = model.predict(img_array)

    # Determine class label based on prediction
    class_label = "cat" if prediction < 0.5 else "dog"
    probability = float(prediction)  # Convert numpy float32 to Python float

    return class_label, probability

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No selected file'})

        try:
            img_path = "F:/cat dog/test_set/test_set/cats/" + file.filename
            class_label, probability = process_image(img_path)
            return render_template('index.html', class_label=class_label, probability=probability, filename=file.filename)
        except Exception as e:
            return jsonify({'error': str(e)})

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

