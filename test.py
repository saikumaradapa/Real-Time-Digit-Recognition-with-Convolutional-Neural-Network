import numpy as np
import cv2
import keras
import cvzone

# Assuming you've already loaded your trained model
model = keras.models.load_model('with_inverted_digit_model.h5')

cap = cv2.VideoCapture(0)  # Using the default camera
cap.set(3,640)
cap.set(4,480)
threshold = 0.8
shade1 = (247,247,67)
shade2 = (109,247,67)
shade3 = (70,255,225)
shade4 = (70,150,255)
orange = (0,69,255)
blue = (255,0,0)
bbox = 220,140, 200,200

while True:
    ret, imgOriginal = cap.read()
    if not ret:
        break

    cvzone.cornerRect(imgOriginal, bbox, l=30, t=5, rt=1,
                      colorR=(255, 0, 255), colorC=(0, 255, 0))

    # Step 1: Correctly slice the region of interest from the original image using bbox
    x, y, w, h = bbox
    img = imgOriginal[y:y + h, x:x + w]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    img = np.asarray(img)
    img = cv2.resize(img, (28, 28))  # Assuming your model takes 28x28 sized images
    img = img / 255.0
    img = img.reshape(1, 28, 28, 1)  # Reshape to feed into the model

    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=1)[0]
    probVal = np.amax(predictions)
    # print(classIndex, probVal)

    # cvzone.putTextRect(imgOriginal, "a real world project by Sai Kumar Adapa",
    #                    (x - 75, y+h + 30), scale=1, thickness=1, colorT=orange,
    #                    colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN,
    #                    offset=12, border=1, colorB=shade3)




    if probVal > threshold:
    # Display the prediction on the original image, slightly above the bounding box
        cvzone.putTextRect(imgOriginal, str(classIndex) + "   " + str(probVal * 100)[:5] + "%",
                           (x+10, y - 20), scale=1.5, thickness=2, colorT=orange,
                           colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN,
                           offset=12, border=1, colorB=shade3)
    else :
        cvzone.putTextRect(imgOriginal, "Place here to find digit",
                           (x+10, y - 20), scale=1.5, thickness=2, colorT=blue,
                           colorR=shade2, font=cv2.FONT_HERSHEY_PLAIN,
                           offset=12, border=1, colorB=shade3)



    cv2.imshow("Original Image", imgOriginal)

    # For visualization purposes
    img = img.squeeze()
    img = (img * 255).astype(np.uint8)
    cv2.imshow("preprocessed", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

