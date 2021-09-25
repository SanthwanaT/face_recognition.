import cv2
import numpy as np 
import face_recognition
# loading the images and converting it into RGB
imgElon = face_recognition.load_image_file('ImageBasic/MUSK1.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImageBasic/Elon-Musk.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

# find faces in the image and find encodings

faceLoc = face_recognition.face_locations(imgElon )[0] # send image (single image only first element)
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]),(faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest )[0] # send image (single image only first element)
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]),(faceLocTest[1], faceLocTest[2]), (255, 0, 255))


#comparing faces and finding the distance between them 

#we use Linear SVM to find whether they match or not 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon ], encodeTest) # lower the distance better the match 
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 2)



cv2.imshow('MUSK', imgElon)
cv2.imshow('MUSKTEST', imgTest)
cv2.waitKey(0) # will display the window infinitely until any keypress (it is suitable for image display).




