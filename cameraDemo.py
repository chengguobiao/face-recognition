#!usr/bin/env python
#encoding:utf-8
from __future__ import division


'''
功能： 人脸识别摄像头视频流数据实时检测模块
'''


import os
import cv2
from faceRegnigtionModel import Model


threshold=0.7  #如果模型认为概率高于70%则显示为模型中已有的人物



def read_name_list(path):
    '''
    读取训练数据集
    '''
    name_list=[]
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class Camera_reader(object):
    def __init__(self):
        self.model=Model()
        self.model.load()
        self.img_size=128


    def build_camera(self):
        '''
        调用摄像头来实时人脸识别
        '''
        face_cascade = cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        name_list=read_name_list('dataset/')
        cameraCapture=cv2.VideoCapture(0)
        success, frame=cameraCapture.read()
        while success and cv2.waitKey(1)==-1:
            success,frame=cameraCapture.read()
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            faces=face_cascade.detectMultiScale(gray, 1.3, 5) 
            for (x,y,w,h) in faces:
                ROI=gray[x:x+w,y:y+h]
                ROI=cv2.resize(ROI, (self.img_size, self.img_size),interpolation=cv2.INTER_LINEAR)
                label,prob=self.model.predict(ROI)  
                if prob>threshold:   
                    show_name=name_list[label]
                else:
                    show_name="Stranger"
                cv2.putText(frame, show_name, (x,y-20),cv2.FONT_HERSHEY_SIMPLEX,1,255,2)  
                frame=cv2.rectangle(frame,(x,y), (x+w,y+h),(255,0,0),2)  
            cv2.imshow("Camera", frame)
        cameraCapture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera=Camera_reader()
    camera.build_camera()


