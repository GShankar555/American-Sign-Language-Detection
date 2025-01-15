import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

TFLITE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "checkpoint", "model.tflite")

IMAGE_SIZE = (160, 160)
CLASS_NAMES = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y",
    "Z", "del", "space",
]

TARGET_FRAME_COUNT = 3
TARGET_CONSECUTIVE_PREDICTIONS = 4
TARGET_PREDICTION_SCORE = 0.92

def load_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter.get_signature_runner("serving_default")

def get_image_array(image_data):
    img_array = tf.keras.utils.img_to_array(image_data)
    img_array = tf.image.resize(img_array, IMAGE_SIZE)  # Ensure resizing to match model's input
    img_array = tf.expand_dims(img_array, 0)
    return img_array

def predict(classify_lite, image_array):
    score_lite = classify_lite(keras_tensor_160=image_array)["output_0"]
    predicted_char = CLASS_NAMES[np.argmax(score_lite)]
    prediction_score = np.max(score_lite)
    return predicted_char, prediction_score

def max_predicted(predictions):
    return max(predictions.items(), key=lambda k: k[1])

def sign_detection(img, classify_lite, previous_predictions, text):
    x1, y1 = 100, 100
    x2, y2 = (x1 + IMAGE_SIZE[0]), (y1 + IMAGE_SIZE[1])

    img = cv2.flip(img, 1)
    img_cropped = img[y1:y2, x1:x2]
    image_data = Image.fromarray(img_cropped)
    image_array = get_image_array(image_data)
    
    predicted_char, prediction_score = predict(classify_lite, image_array)
    
    # Only update predictions if the score meets the threshold
    if prediction_score >= TARGET_PREDICTION_SCORE:
        previous_predictions[predicted_char] += 1
    else:
        predicted_char = ""  # Display nothing if the score is below threshold
    
    letter, count = max_predicted(previous_predictions)
    
    if count >= TARGET_CONSECUTIVE_PREDICTIONS:
        previous_predictions = {letter: 0 for letter in CLASS_NAMES}  # Reset predictions
        if letter == "space":
            text += " "
        elif letter == "del":
            text = text[:-1]
        else:
            text += letter

    if prediction_score >= 0.7:
        cv2.putText(img, predicted_char.upper(), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)
    
    cv2.putText(img, f"(score = {prediction_score:.2f})", (5, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return img, predicted_char, prediction_score, text, previous_predictions
