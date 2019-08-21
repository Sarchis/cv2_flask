from flask import Flask, render_template, Response
import cv2
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)
video = cv2.VideoCapture(0)

#opencv initialization
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')

emotions = ('Enojado', 'Disgustado', 'Asustado', 'Feliz', 'Triste', 'Sorprendido', 'Neutral')
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list = ['Male', 'Female']


def initialize_caffe_model():
    print('Loading models...')
    age_net = cv2.dnn.readNetFromCaffe(
                        "./age_gender_models/deploy_age.prototxt",
                        "./age_gender_models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe(
                        "./age_gender_models/deploy_gender.prototxt",
                        "./age_gender_models/gender_net.caffemodel")

    return (age_net, gender_net)

def gen():
    from keras.models import model_from_json
    #-----------------------------
    # face expression recognizer initialization
    model = model_from_json(open("./model/facial_expression_model_structure.json", "r").read())
    model.load_weights('./model/facial_expression_model_weights.h5') #load weights
    #-----------------------------

    """Video streaming generator function."""
    while True:
        rval, frame = video.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # print(faces)

        for (x,y,w,h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image

            detected_face = frame[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            predictions = model.predict(img_pixels)

            max_index = np.argmax(predictions[0])

            emotion = emotions[max_index]
            # return emotion
            print(emotion)

        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

@app.route("/", defaults={"js": "video"})
@app.route("/<any(video):js>")
# @app.route('/')
def index(js):
    # return 'Hello, World!'
    # return render_template('index.html')
    expression = gen()
    print('expression ', expression)
    # expression = 'Happy'
    return render_template("{0}.html".format(js), js=js, expression=expression)

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8080')
