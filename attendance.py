import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#if we had hundreds of pics then we could have created a loop and load all the images using the OS, which is used to access the files

ronaldo_image = face_recognition.load_image_file("photos/ronaldo.jpg")      
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]         #creating variables

messi_image = face_recognition.load_image_file("photos/messi.jpg")
messi_encoding = face_recognition.face_encodings(messi_image)[0]

neymar_image = face_recognition.load_image_file("photos/neymar.jpg")
neymar_encoding = face_recognition.face_encodings(neymar_image)[0]

haaland_image = face_recognition.load_image_file("photos/haaland.jpg")
haaland_encoding = face_recognition.face_encodings(haaland_image)[0]

mbappe_image = face_recognition.load_image_file("photos/mbappe.jpg")
mbappe_encoding = face_recognition.face_encodings(mbappe_image)[0]

known_face_encoding = [
    ronaldo_encoding,
    messi_encoding,         #list for encoding
    neymar_encoding,
    haaland_encoding,
    mbappe_encoding
]

known_faces_names = [
    "ronaldo",
    "messi",
    "neymar",               #list for names
    "haaland",
    "mbappe"
]

students = known_faces_names.copy()
#some more variables
face_locations = []  #face co ordinates
face_encodings = []  #the raw data
face_names = []       #name of the face if present in list
s = True 

now = datetime.now()  #to get the exact date and time
current_date = now.strftime("%Y-%m-%d") 

f = open(current_date+'.csv','w+',newline='')   #using the open method with the parameters
#we are opening it with w+ method or write plus
lnwriter = csv.writer(f)   #writer class instance, we will be using it when we write the data in th csv file

while True:  #infinite loop
    _,frame = video_capture.read()     #reading the video input, then by the read method we are extracting the video data
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)   #decreasing the size of input coming from web cam
    rgb_small_frame = small_frame[:,:,::-1]  #cv2 takes the input in bgr format so we are converting it to rgb

    if s:   #if true
        face_locations = face_recognition.face_locations(rgb_small_frame)  #if there is a face or not
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)  #store face data of coming frame
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)  #used numpy to get the best fit
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]  #if the face exists, the name variable will have the name of the face
 
            face_names.append(name)    #append the face name in the csv file
            if name in known_faces_names:
                if name in students:
                    students.remove(name)   #we are removing the name so as to avoid entering the name multiple times, as when the student comes in front of the camera, multiple frames are taken.
                    print(students)
                    current_time = now.strftime("%H-%M-%S")  #strftime comverts the daye and time to their string representation
                    lnwriter.writerow([name,current_time])
    cv2.imshow("attedance system",frame)   #show the output to the user(openCV) parameters given are, first the text that will be present in the GUI, second is the video
    if cv2.waitKey(1) & 0xFF == ord('q'):   #exit condition, it will be executed when we press the 'q' button
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()    #closing a csv file