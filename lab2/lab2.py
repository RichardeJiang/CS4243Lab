import cv2
import os
import numpy as np

def convertBGRToHSV(picPath, title):
	image = cv2.imread(picPath)
	#print len(image) #vertical pixels
	#print len(image[0]) #horizontal pixels
	#print len(image[0][0]) #BGR
	HArray = np.ndarray(shape=(len(image), len(image[0])))
	SArray = np.ndarray(shape=(len(image), len(image[0])))
	VArray = np.ndarray(shape=(len(image), len(image[0])))

	rowIndex = 0
	colIndex = 0
	for row in image:
		colIndex = 0
		for cell in row:
			B = cell[0]
			G = cell[1]
			R = cell[2]
			Bp = B / 255
			Gp = G / 255
			Rp = R / 255
			Cmax = max (Bp, Gp, Rp)
			Cmin = min (Bp, Gp, Rp)
			delta = Cmax - Cmin

			if (delta == 0):
				H = 0
			elif (Cmax == Rp):
				H = (((Gp - Bp) / delta) % 6) * 60
			elif (Cmax == Gp):
				H = ((Bp - Rp) / delta + 2) * 60
			else:
				H = ((Rp - Gp) / delta + 4) * 60

			if (Cmax == 0):
				S = 0
			else:
				S = delta / Cmax

			V = Cmax

			HArray[rowIndex][colIndex] = H
			SArray[rowIndex][colIndex] = S
			VArray[rowIndex][colIndex] = V

			colIndex += 1

		rowIndex += 1
	
	cv2.imwrite('output/' + '_hue.jpg', HArray)
	cv2.imwrite('output/' + '_saturation.jpg', SArray)
	cv2.imwrite('output/' + '_value.jpg', VArray)
	return

def convertHSVToBGR(picPath, title):
	return

def writeImage(picPath):
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
			title = picName.split('.')[0]
			convertBGRToHSV('lab2_pictures/' + picName, title)
			convertHSVToBGR('lab2_pictures/' + picName, title)
			break

	else:
		print 'No picture folder found in current directory!\n'
	pass