#!usr/bin/env python
#encoding:utf-8
from __future__ import division


'''
功能： 图像的数据预处理、标准化部分
'''


import os
import cv2
import time



def readAllImg(path,*suffix):
    '''
    基于后缀读取文件
    '''
    try:
        s=os.listdir(path)
        resultArray = []
        fileName = os.path.basename(path)
        resultArray.append(fileName)
        for i in s:
            if endwith(i, suffix):
                document = os.path.join(path, i)
                img = cv2.imread(document)
                resultArray.append(img)
    except IOError:
        print("Error")

    else:
        print("读取成功")
        return resultArray


def endwith(s,*endstring):
    '''
    对字符串的后续和标签进行匹配
    '''
    resultArray = map(s.endswith,endstring)
    if True in resultArray:
        return True
    else:
        return False


def readPicSaveFace(sourcePath,objectPath,*suffix):
    '''
    图片标准化与存储
    '''
    if not os.path.exists(objectPath):
        os.makedirs(objectPath)
    try:
        resultArray=readAllImg(sourcePath,*suffix)
        count=1
        face_cascade=cv2.CascadeClassifier('config/haarcascade_frontalface_alt.xml')
        for i in resultArray:
            if type(i)!=str:
              gray=cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
              faces=face_cascade.detectMultiScale(gray, 1.3, 5)
              for (x,y,w,h) in faces:
                listStr=[str(int(time.time())),str(count)]  
                fileName=''.join(listStr)
                f=cv2.resize(gray[y:(y+h),x:(x+w)],(200, 200))
                cv2.imwrite(objectPath+os.sep+'%s.jpg' % fileName, f)
                count+=1
    except Exception as e:
        print("Exception: ",e)
    else:
        print('Read  '+str(count-1)+' Faces to Destination '+objectPath)

if __name__ == '__main__':
    print('dataProcessing!!!')
    readPicSaveFace('data/KA/','dataset/KA/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/KL/','dataset/KL/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/KM/','dataset/KM/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/KR/','dataset/KR/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/MK/','dataset/MK/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/NA/','dataset/NA/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/NM/','dataset/NM/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/TM/','dataset/TM/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/UY/','dataset/UY/','.jpg','.JPG','png','PNG','tiff')
    readPicSaveFace('data/YM/','dataset/YM/','.jpg','.JPG','png','PNG','tiff')

