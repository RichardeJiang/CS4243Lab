import cv2
import os
import numpy as np

def convertBGRToHSV(picPath):
	image = cv2.imread(picPath)
	#print len(image) #vertical pixels
	#print len(image[0]) #horizontal pixels
	#print len(image[0][0]) #BGR
	print image
	hsv = cv2.cvtColor(image, cv2.cv.CV_BGR2HSV)
	print hsv
	cv2.imwrite('test1.jpg', hsv)
	image2 = cv2.cvtColor(hsv, cv2.cv.CV_HSV2BGR)
	cv2.imwrite('test2.jpg', image2)
	return

def convertHSVToBGR(picPath):
	return

if (__name__=='__main__'):
	fileList = os.listdir('.')
	if (any('lab2_pictures' in fileName for fileName in fileList)):
		dirName = os.listdir('lab2_pictures/')
		picList = []
		for picName in dirName:
			if (picName.endswith('.jpg')):
				picList.append(picName)
		
		for picName in picList:
			convertBGRToHSV('lab2_pictures/' + picName)
			convertHSVToBGR('lab2_pictures/' + picName)
			break

	else:
		print 'No picture folder found in current directory!\n'
	pass