import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from time import sleep
face_classifier = cv2.CascadeClassifier(r'D:\pds_project\haarcascade_frontalface_default.xml')
model =load_model('D:/projects/Age_Gender_Classsification.h5')
gender_dict={0:'Male',1:'Female'}


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        #print(roi_gray.shape)
        roi_gray = cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
        print(roi_gray.shape)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)


        
            #pred = model.predict(roi_gray.reshape(1, 128, 128, 1))
            pred = model.predict(roi)
            print(pred)
            pred_gender = gender_dict[round(pred[0][0][0])]
            pred_age = round(pred[1][0][0])
            #print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
            label_position1 = (x,y-10)
            cv2.putText(frame,pred_gender,label_position1,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            label_position2 = (x+w-50,y-10)
            cv2.putText(frame,str(pred_age),label_position2,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)



    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
