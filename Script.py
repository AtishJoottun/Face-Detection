import cv2 as cv

vid = cv.VideoCapture(0)

# Defines the data sets
face_classifier = cv.CascadeClassifier(cv.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_classifier = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

while (True):
    #reads the camera module
    ret, Frame = vid.read()

    # Turns colour to gray scale
    gray_image = cv.cvtColor(Frame, cv.COLOR_BGR2GRAY)
    #detect the faces in the gray scale
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draws the box
        cv.rectangle(Frame, (x, y), (x+w, y+h), (0, 255, 0), 1)

        # Extract the faces from the frame
        gray_roi = gray_image[y:y+h, x:x+w]
        Colour_roi = Frame[y:y+h, x:x+w]

        Eyes = eye_classifier.detectMultiScale(gray_roi)
        # Draws the points for the eyes
        for(ex, ey, ew, eh) in Eyes:
            cv.rectangle(Colour_roi, (int(ex + ew/2), int(ey + eh/2)), (int(ex + ew/2), int(ey + eh/2)), (0,0,255),5)

    # Display the frames
    cv.imshow('frame', Frame)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

vid.release()
cv.destroyAllWindows()
