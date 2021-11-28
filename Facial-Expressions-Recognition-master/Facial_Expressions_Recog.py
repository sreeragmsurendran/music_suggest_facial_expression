from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Users\91623\Downloads\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\91623\Downloads\Facial-Expressions-Recognition-master\Facial-Expressions-Recognition-master\Emotion_little_vgg.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']

cap = cv2.VideoCapture(0)



while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    # rect,face,image = face_detector(frame)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            #music player

            from pygame import mixer
            if label =='Happy':

                mixer.init()  # Initialzing pyamge mixer

                mixer.music.load('angrysong.mp3')  # Loading Music File

                mixer.music.play()  # Playing Music with Pygame
            elif label =='Angry':
                mixer.init()  # Initialzing pyamge mixer

                mixer.music.load('happysong.mp3')  # Loading Music File

                mixer.music.play()  # Playing Music with Pygame
            elif label == 'Neutral':
                mixer.init()  # Initialzing pyamge mixer

                mixer.music.load('angrysong.mp3')  # Loading Music File

                mixer.music.play()  # Playing Music with Pygame
            elif label == 'Sad':
                mixer.init()  # Initialzing pyamge mixer

                mixer.music.load('happysong.mp3')  # Loading Music File

                mixer.music.play()  # Playing Music with Pygame
            elif label == 'Surprise':
                mixer.init()  # Initialzing pyamge mixer

                mixer.music.load('happysong.mp3')  # Loading Music File

                mixer.music.play()  # Playing Music with Pygame
            else :
                break

            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
    cv2.imshow('Emotion Detector',frame)
    bool = False
    def mouse_click(event, x, y,
                flags, param):
        print("function worked")
        if event == cv2.EVENT_LBUTTONDOWN:
            # global bool
            # bool = True
            # print("event worked")
            cv2.destroyAllWindows()
        if event == cv2.EVENT_RBUTTONDOWN:
            import sys
            sys.exit()

    cv2.setMouseCallback('Emotion Detector', mouse_click)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    print("expression")
    # if bool== True :
    #     break
    cv2.waitKey(0)
    # if bool == False:
    #     break;


# cap.release()
cv2.destroyAllWindows()


























