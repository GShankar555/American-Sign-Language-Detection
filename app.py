from flask import Flask, render_template, Response
import cv2
from utils import sign_detection, load_model, CLASS_NAMES

app = Flask(__name__)

def generate_frames():
    camera = cv2.VideoCapture(0)
    classify_lite = load_model()
    previous_predictions = {letter: 0 for letter in CLASS_NAMES}
    text = ""
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame, letter, prediction_score, text, previous_predictions = sign_detection(
                frame, classify_lite, previous_predictions, text
            )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

@app.route('/')
def about():
    return render_template('dataset.html')

@app.route('/detection')
def index():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)
