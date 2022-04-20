# imports
from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import os 
import numpy as np

# argparse stuff here
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="enter path to output directory")
args = vars(ap.parse_args())
args["cascade"] = "data/haarcascades/haarcascade_frontalface_default.xml"


# init detector
detector = cv2.CascadeClassifier(args["cascade"])

# handling paths
if not os.path.exists(os.path.sep.join([args["output"]])):
	os.makedirs(os.path.sep.join([args["output"]]))

# starting the thingamajig
print("\nStarting up webcam stream..\n")
stream = VideoStream(src=0).start()

time.sleep(2.0)
total = 0

# keep capturing frames
while True:

	frame = stream.read()
		
	frame = imutils.resize(frame, width=400)

	# get all detected faces
	detected = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
	boxed = np.array([])

	# display frames
	cv2.imshow("Mirror x2 On The Wall", frame)
	key = cv2.waitKey(1) & 0xFF

	# quit
	if key == ord('q'):
		break

	# capture image
	elif key == ord('c'):
		p = os.path.sep.join([args["output"], "{}.png".format(str(total + 1).zfill(5))])

		# taking coordinates of detected face
		# note, this will break if you have multiple faces. You can loop to handle those.
		for (x, y, w, h) in detected:
			boxed = frame[y:y+h, x:x+w]

		# writing face in grayscale IF a face has been detected
		if boxed.any() and detected.any():
			cv2.imwrite("images/", cv2.cvtColor(boxed, cv2.COLOR_BGR2GRAY))

		else:
			continue
		
		# show image that has been captured
		cv2.imshow("image {}".format(int(total + 1)), cv2.cvtColor(boxed, cv2.COLOR_BGR2GRAY))
		cv2.waitKey(3)

		total += 1


print("{} face dataset captured and stored".format(total))
print("\nWe're done!\n")
cv2.destroyAllWindows()
stream.stop()