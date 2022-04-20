# imports
import os
import time
import cv2
import imutils
import numpy as np
from imutils.video import VideoStream

# detector
detector = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml');

# IMAGE Directory
CUR_DIR = os.getcwd()
IMAGE_DIR = os.path.join(CUR_DIR, 'images')
USER = input("Enter Your Name : ")
USER_DIR = os.path.join(IMAGE_DIR, USER)

if not os.path.exists(USER_DIR):
    os.makedirs(USER_DIR)
    print("Your Image Folder has been Created! :", USER_DIR)

# starting the
print("\nStarting up webcam stream..\n")
stream = VideoStream(src=0).start()

time.sleep(2.0)
total = 0

# keep capturing frames
while True:
    frame = stream.read()
    frame = imutils.resize(frame, width=400)
    # get all detected faces
    detected = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    boxed = np.array([])
    # display frames
    cv2.imshow("Mirror x2 On The Wall", frame)
    key = cv2.waitKey(1) & 0xFF
    # quit
    if key == ord('q'):
        break
    # capture image
    elif key == ord('c'):
        p = os.path.sep.join([USER_DIR, "{}.png".format(str(total + 1))])

        # taking coordinates of detected face
        # note, this will break if you have multiple faces. You can loop to handle those.
        for (x, y, w, h) in detected:
            boxed = frame[y:y + h, x:x + w]

        # writing face in grayscale IF a face has been detected
        if boxed.any() and detected.any():
            cv2.imwrite(p, cv2.cvtColor(boxed, cv2.COLOR_BGR2GRAY))

        else:
            continue

        # show image that has been captured
        cv2.imshow("image {}".format(int(total + 1)), cv2.cvtColor(boxed, cv2.COLOR_BGR2GRAY))
        cv2.waitKey(3)

        total += 1

print("{} face dataset captured and stored".format(total))
cv2.destroyAllWindows()
stream.stop()