Title
Digit Recognition from Live Video Stream

Description
A real-time digit recognition system that captures video from a device's camera, processes the stream frame-by-frame, and predicts the handwritten digit present in a designated rectangular region on the screen. Leveraging the power of deep learning, the system predicts the digit and displays the result, all in real time.

Tech Stack
Machine Learning Framework: Keras (with TensorFlow backend)
Image Processing: OpenCV
Web Framework: Flask
UI Components: HTML, CSS, JavaScript
Others: cvzone (for additional computer vision utilities)
Data Set
The project uses the popular MNIST dataset, which is a collection of handwritten digits. Each image in the dataset is a 28x28 grayscale image representing a digit from 0 to 9.

Features
Real-time Prediction: Predicts handwritten digits from a live video stream.
Bounded Region of Interest: A designated rectangular area on the video feed where the handwritten digit is placed for recognition.
Accuracy Threshold: Only displays predictions that exceed a certain confidence threshold.
Web Interface: An interactive user interface hosted on a web server, allowing for easy access from various devices.
Learning Points
Accuracy Challenges: Initially faced challenges with recognition accuracy, especially in real-world lighting conditions.
Image Inversion for Improved Accuracy: By training the model with both regular and inverted images (black digit on white background and vice-versa), the system's accuracy significantly improved. This helped in adapting the model to various writing instruments and surfaces.
Real-world Application: The inclusion of inverted images in the dataset made the model robust, transforming it from just a prototype to a real-world working application.
Installation and Setup
Clone the repository.
Install required Python libraries/packages: pip install -r requirements.txt
Run the Flask server: python app.py
Access the application on your browser at http://127.0.0.1:5000.
