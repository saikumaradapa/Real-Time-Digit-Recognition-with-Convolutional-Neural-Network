# Importing necessary libraries and modules
from flask import Flask, render_template, Response
import cv2
import numpy as np
import cvzone
import keras

# Initializing Flask app
app = Flask(__name__)

# Load pre-trained model for digit prediction
model = keras.models.load_model('with_inverted_digit_model.h5')

# Set up video capture from default camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set video width
cap.set(4, 480)  # Set video height

# Threshold for model prediction probability
threshold = 0.8

# Predefined color shades for displaying text and bounding box
shade1 = (247, 247, 67)
shade2 = (109, 247, 67)
shade3 = (70, 255, 225)
shade4 = (70, 150, 255)
orange = (0, 69, 255)
blue = (255, 0, 0)

# Bounding box coordinates where digit should be placed for prediction
bbox = 220, 140, 200, 200


def gen_frames():
    """Generate frames from the webcam feed and predict digits from the specified bounding box."""
    while True:
        success, imgOriginal = cap.read()
        if not success:
            break
        else:
            # Draw a rectangular box to place the digit
            cvzone.cornerRect(imgOriginal, bbox, l=30, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0))

            # Extract the region of interest based on bounding box
            x, y, w, h = bbox
            img = imgOriginal[y:y + h, x:x + w]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

            # Preprocess the image for prediction
            img = np.asarray(img)
            img = cv2.resize(img, (28, 28))
            img = img / 255.0
            img = img.reshape(1, 28, 28, 1)

            # Make prediction
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=1)[0]
            probVal = np.amax(predictions)

            # Display the prediction on the live webcam feed
            if probVal > threshold:
                cvzone.putTextRect(imgOriginal, f"{classIndex}   {probVal * 100:.2f}%",
                                   (x + 10, y - 20), scale=1.5, thickness=2, colorT=orange,
                                   colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN, offset=12, border=1, colorB=shade3)
            else:
                cvzone.putTextRect(imgOriginal, "Place here to find digit",
                                   (x + 10, y - 20), scale=1.5, thickness=2, colorT=blue,
                                   colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN, offset=12, border=1, colorB=shade3)

        ret, buffer = cv2.imencode('.jpg', imgOriginal)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    """Return the main HTML page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video feed route for the live webcam feed."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
