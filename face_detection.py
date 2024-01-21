import cv2
import numpy as np
import face_recognition

imElon = face_recognition.load_image_file('face_detection/images/Elon_Musk_1.jpg')
imElon = cv2.cvtColor(imElon,cv2.COLOR_BGR2RGB)

imTest = face_recognition.load_image_file('face_detection/images/Elon_Musk_2.jpg')
imTest = cv2.cvtColor(imTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imElon)[0]
encodeElon = face_recognition.face_encodings(imElon)[0]
cv2.rectangle(imElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imTest)[0]
encodeTest = face_recognition.face_encodings(imTest)[0]
cv2.rectangle(imTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Elon Musk',imElon)
cv2.imshow('Elon Test',imTest)
cv2.waitKey(0)