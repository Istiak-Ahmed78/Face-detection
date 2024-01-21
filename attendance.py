import cv2
import os
import face_recognition
import numpy as np


path = 'images'
images = []
classNames = []
dirList = os.listdir(path)

for cl in dirList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncoding(images):
    encodedList=[]
    for i in images:
        im = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(im)[0]
        encodedList.append(encode)
    return encodedList


knownFaces = findEncoding(images)
print(len(knownFaces))
cap = cv2.VideoCapture(0)


while True:
    success, img = cap.read()
    if(not cap.isOpened()):
        print('Camera is not found. Actually not opening.')
        break
    imS = cv2.resize(img, (0,0),None,0.25,0.25)
    imS = cv2.cvtColor(imS,cv2.COLOR_BGR2RGB)

    faceLocations = face_recognition.face_locations(imS)
    faceEncodings = face_recognition.face_encodings(imS,faceLocations)
    print(f'We have detected {len(faceLocations)} faces')
    if len(faceLocations) != 0:
        for  fEncoding,fLocation in zip(faceEncodings,faceLocations):
            matches = face_recognition.compare_faces(knownFaces, fEncoding)
            faceDis = face_recognition.face_distance(knownFaces, fEncoding)
            matchedIndex = np.argmin(faceDis)
            print(matches)
            if matches[matchedIndex]:
                print(f"It is known person")
                name = classNames[matchedIndex].upper()
                # Draw rect
                y1,x1,y2,x2 = fLocation
                y1,x1,y2,x2 = y1*4,x1*4,y2*4,x2*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)


    k = cv2.waitKey(30) & 0xff
    if(k==27):
        break
    cv2.imshow('Webcame', img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
