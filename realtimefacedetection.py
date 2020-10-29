# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 08:51:30 2020

@author: Lenovo
"""


import cv2
import face_recognition

webcam_video_stream=cv2.VideoCapture(0)
all_face_locations=[]

while True:
    r,current_frame=webcam_video_stream.read()
    current_frame_small=cv2.resize(current_frame,(0,0),fx=0.25,fy=0.25)
    #no_of_times_to_upsample
    all_face_locations=face_recognition.face_locations(current_frame_small,number_of_times_to_upsample=2,model='hog')
    for index,current_face_locations in enumerate(all_face_locations):
        top_pos,right_pos,bottom_pos,left_pos=current_face_locations
        top_pos=top_pos*4
        right_pos=right_pos*4
        bottom_pos=bottom_pos*4
        left_pos=left_pos*4
        print("Face found {} at top:{},right:{},bottom:{},left:{}".format(index+1,top_pos,right_pos,bottom_pos,left_pos))
        current_face_image=current_frame[top_pos:bottom_pos,left_pos:right_pos]
        cv2.rectangle(current_frame,(left_pos,top_pos),(right_pos,bottom_pos),(0,0,255),2)
        current_frame[top_pos:bottom_pos,left_pos:right_pos]=current_face_image
    cv2.imshow("Webcam video",current_frame)
        
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    
webcam_video_stream.release()
cv2.destroyAllWindows()
