from tensorflow.keras.models import model_from_json
import model_functions
from numpy import argmax

def load_model():
    with open("model/test_model/model_arch.json", 'r') as f:
        model = model_from_json(f.read())

    model.load_weights('model/test_model/my_model_weights.h5')
    return model

def predict():
    # Image after pre-processing and in 2D format
    result_image = model_functions.pre_processImage()
    model = load_model()
    pred = model.predict(result_image)

    pred = argmax(pred, axis=1)
    pred = pred.astype(str)

    pred[pred == '0'] = 'G'
    pred[pred == '1'] = 'A'
    pred[pred == '2'] = 'O'

    return pred[0]
