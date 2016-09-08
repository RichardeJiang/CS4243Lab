import cv2
import numpy as np
import os

def MyConvolve(image, ff):
	result = np.zeros(image.shape)
	if (ff == 1):
		kernelV = np.array([(-1, 0, 1), (-1, 0, 1), (-1, 0, 1)])
		kernelH = np.array([(1, 1, 1), (0, 0, 0), (-1, -1, -1)])
	else:
		kernelV = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
		kernelH = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
	rows = len(image)
	cols = len(image[0])

	for row in range(1, rows-1):
		for col in range(1, cols):
			currWindow = image[row-1:row+2, col-1:col+2]
			result[row][col] = currWindow * kernelH

	return result

def edgeDetection(fileName):
	image = cv2.imread(fileName, 0)
	PrewittResult = MyConvolve(image, 1)
	SobelResult = MyConvolve(image, 2)
	title = fileName.split('.')[0]
	return

def edgeThinning(fileName):
	image = cv2.imread(fileName, 0)
	title = fileName.split('.')[0]
	return

if (__name__=="__main__"):
	fileList = os.listdir('.')
	for fileName in fileList:
		if fileName.endswith('.jpg') and ('test' in fileName):
			title = fileName.split('.')[0]
			image = cv2.imread(fileName, 0)
			edgeDetection(image, title)
			edgeThinning(image, title)
	pass