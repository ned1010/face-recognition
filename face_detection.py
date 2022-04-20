import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("./data/face-trainer.yml")

labels = {"person_name": 1}
with open("data/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}


cap = cv2.VideoCapture(0)

while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        # recognize? deep learned model predict keras tensorflow pytorch scikit learn
        id, conf = recognizer.predict(roi_gray)

        if conf >= 4 and conf <= 100:
            print(conf)
            # print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id]
            color = (255, 255, 255)
            stroke = 2
            conf = "  {0}%".format(round(100 - conf))
            #name_conf = str(name) + str(conf)
            #print(name_conf)

            cv2.putText(frame, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(conf), (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            #cv2.putText(frame, name_conf, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "Unknown", (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        color = (255, 0, 0)  # BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
