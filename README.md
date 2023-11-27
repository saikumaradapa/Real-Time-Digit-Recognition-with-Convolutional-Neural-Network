
# A Real World Hand Written Digit Recognition from Live Video Stream

**Description :** 
A real-time digit recognition system that captures video from a device's camera, processes the stream frame-by-frame, and predicts the handwritten digit present in a designated rectangular region on the screen. Leveraging the power of deep learning, the system predicts the digit and displays the result, all in real time.

**Tech Stack :** </br>
**Machine Learning Framework**: Keras (with TensorFlow backend) </br>
**Image Processing**: OpenCV, cvzone </br>
**Web Framework**: Flask </br>
**UI Components**: HTML, CSS, JavaScript </br>
**Others**: cvzone (for additional computer vision utilities) </br>

**Data Set :** </br>
The project uses the popular MNIST dataset, which is a collection of handwritten digits. Each image in the dataset is a 28x28 grayscale image representing a digit from 0 to 9. </br>

**Features :** </br>
**Real-time Prediction**: Predicts handwritten digits from a live video stream. </br>
**Bounded Region of Interest**: A designated rectangular area on the video feed where the handwritten digit is placed for recognition. </br>
**Accuracy Threshold**: Only displays predictions that exceed a certain confidence threshold. </br>
**Web Interface**: An interactive user interface hosted on a web server, allowing for easy access from various devices. </br>

**Learning Points :**  </br>
**Accuracy Challenges**: Initially faced challenges with recognition accuracy, especially in real-world lighting conditions. </br>
**Image Inversion for Improved Accuracy**: By training the model with both regular and inverted images (black digit on white background and vice-versa), the system's accuracy significantly improved. This helped in adapting the model to various writing instruments and surfaces. </br>
**Real-world Application**: The inclusion of inverted images in the dataset made the model robust, transforming it from just a prototype to a real-world working application. </br>

**Installation and Setup :**  </br>
Clone the repository.</br>
Install required Python libraries/packages: pip install -r requirements.txt </br>
Run the Flask server: python app.py </br>
Access the application on your browser at http://127.0.0.1:5000. </br>







https://github.com/saikumaradapa/Real-Time-Digit-Recognition-with-Convolutional-Neural-Network/assets/96902883/93a0b5c4-2e8d-4926-86a5-1e1183c9899d


