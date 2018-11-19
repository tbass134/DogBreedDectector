import sys
import os
import shutil
import time
import traceback
import requests

import numpy as np

from flask import Flask, request, jsonify
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# inputs
num_classes = 120
im_size = 299

df = pd.read_csv('labels.csv')
sorted_breeds_list = sorted(list(df.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index))

model = load_model('2018-11-15_dog_breed_model.h5')
graph = tf.get_default_graph()

def predict_from_image(img_path):

    img = image.load_img(img_path, target_size=(im_size, im_size))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    global graph
    with graph.as_default():
        pred = model.predict(img_tensor)
        predicted_class = sorted_breeds_list[np.argmax(pred)]

    return predicted_class

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json = request.json
        print(json)

        image_path = json['image_path']
        ts = time.gmtime()
        ts_str = time.strftime("%s", ts)
        filename = ts_str+".jpg"
        f = open(filename,'wb')
        f.write(requests.get(image_path).content)
        f.close()

        prediction = predict_from_image(filename)
        os.remove(filename)
        print("File Removed!")

        print("prediction: {}".format(prediction))

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def setup():
    return
setup()

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
