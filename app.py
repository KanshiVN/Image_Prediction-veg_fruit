import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model(r'C:\Users\HP\Desktop\DEEP learning\Image_classify.keras')

data_cat = ['apple','banana','beetroot','bell pepper','cabbage','capsicum',
             'carrot','cauliflower','chilli pepper','corn','cucumber','eggplant',
             'garlic','ginger','grapes','jalepeno','kiwi','lemon','lettuce','mango',
             'onion','orange','paprika','pear','peas','pineapple','pomegranate',
             'potato','raddish','soy beans','spinach','sweetcorn','sweetpotato',
             'tomato','turnip','watermelon']

img_height, img_width = 180, 180
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    image = tf.keras.utils.load_img(filepath, target_size=(img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, 0)

    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])
    label = data_cat[np.argmax(score)]
    accuracy = np.max(score) * 100

    return render_template('index.html', 
                           label=label, 
                           accuracy=accuracy,
                           image=filepath)

if __name__ == '__main__':
    app.run(debug=True)
