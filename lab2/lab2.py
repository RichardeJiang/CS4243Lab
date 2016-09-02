import cv2
import os
import numpy as np
import math

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
			Bp = float(B) / 255
			Gp = float(G) / 255
			Rp = float(R) / 255
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

			HArray[rowIndex][colIndex] = H / 360 * 255
			SArray[rowIndex][colIndex] = S * 255  #from cvtColor() doc, it should be 255*S
			VArray[rowIndex][colIndex] = V * 255  #otherwise the output will be dark

			colIndex += 1

		rowIndex += 1
	
	cv2.imwrite('output/' + title + '_hue.jpg', HArray)
	cv2.imwrite('output/' + title + '_saturation.jpg', SArray)
	cv2.imwrite('output/' + title + '_brightness.jpg', VArray)
	return

def convertHSVToBGR(title):
	try:
		HArray = cv2.imread('output/' + title + '_hue.jpg')
		SArray = cv2.imread('output/' + title + '_saturation.jpg')
		VArray = cv2.imread('output/' + title + '_brightness.jpg')
	except:
		print 'Target files not found in current directory!\n'
	else:
		resultArray = convertHSVToBGRCalculation(HArray, SArray, VArray)
		cv2.imwrite('output/' + title + '_hsv2rgb.jpg', resultArray)
	return

def convertHSVToBGRCalculation(HArray, SArray, VArray):
	rows = len(HArray)
	cols = len(HArray[0])
	resultArray = np.ndarray(shape=(rows, cols, 3))
	for row in range(0, rows):
		for col in range(0, cols):
			H = float(HArray[row][col][0]) / 255 * 360
			S = float(SArray[row][col][0]) / 255
			V = float(VArray[row][col][0]) / 255
			C = S * V
			X = C * (1 - abs((H / 60) % 2 - 1))
			m = V - C
			if (H in range(0, 60)):
				Rp, Gp, Bp = C, X, 0
			elif (H in range(60, 120)):
				Rp, Gp, Bp = X, C, 0
			elif (H in range(120, 180)):
				Rp, Gp, Bp = 0, C, X
			elif (H in range(180, 240)):
				Rp, Gp, Bp = 0, X, C
			elif (H in range(240, 300)):
				Rp, Gp, Bp = X, 0, C
			else:
				Rp, Gp, Bp = C, 0, X
			resultArray[row][col][0] = (Bp + m) * 255
			resultArray[row][col][1] = (Gp + m) * 255
			resultArray[row][col][2] = (Rp + m) * 255

	return resultArray

def histogramEq(title):
	fileList = os.listdir('output/')
	try:
		HArray = cv2.imread('output/' + title + '_hue.jpg')
		SArray = cv2.imread('output/' + title + '_saturation.jpg')
		VArray = cv2.imread('output/' + title + '_brightness.jpg')
	except:
		print 'Target files not found in current directory!\n'
	else:
		histogram = np.zeros(256)
		sumPlot = np.zeros(256)
		rows = len(VArray)
		cols = len(VArray[0])
		newVArray = np.ndarray(shape=(rows, cols, 1)) #3D array is required because imwrite will convert to 3D
		sumOfHis = 0
		for row in range(0, rows):
			for col in range(0, cols):
				histogram[int(VArray[row][col][0])] += 1

		for index in range(0, 256):
			sumOfHis += histogram[index]
			sumPlot[index] = sumOfHis

		unit = float(sumOfHis) / 256
		for row in range(0, rows):
			for col in range(0, cols):
				newVArray[row][col][0] = math.floor(sumPlot[int(VArray[row][col][0])] / unit)
		
		resultArray = convertHSVToBGRCalculation(HArray, SArray, newVArray)
		cv2.imwrite('output/' + title + '_histeq.jpg', resultArray)
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
			convertHSVToBGR(title)
			histogramEq(title)

	else:
		print 'No picture folder found in current directory!\n'
	pass