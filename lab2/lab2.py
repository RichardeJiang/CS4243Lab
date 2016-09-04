import cv2
import os
import numpy as np
import math

def convertBGRToHSV(title):
	try:
		image = cv2.imread('lab2_pictures/' + title + '.jpg')
	except:
		print 'Target files not found in current directory!\n'
	else:

		#print len(image) #vertical pixels
		#print len(image[0]) #horizontal pixels
		#print len(image[0][0]) #BGR
		HArray = np.ndarray(shape=(len(image), len(image[0])), dtype=np.uint8)
		SArray = np.ndarray(shape=(len(image), len(image[0])), dtype=np.uint8)
		VArray = np.ndarray(shape=(len(image), len(image[0])), dtype=np.uint8)

		rows = len(image)
		cols = len(image[0])
		for row in range(0, rows):
			for col in range(0, cols):
				B = image[row][col][0]
				G = image[row][col][1]
				R = image[row][col][2]
				Bp = float(B) / 255.0
				Gp = float(G) / 255.0
				Rp = float(R) / 255.0
				Cmax = max (Rp, Gp, Bp)
				Cmin = min (Rp, Gp, Bp)
				delta = Cmax - Cmin

				if (delta == 0):
					H = 0
				elif (Cmax == Rp):
					H = (int((Gp - Bp) / delta) % 6) * 60
					#H = ((Gp - Bp) / delta) * 60
				elif (Cmax == Gp):
					H = (((Bp - Rp) / delta) + 2) * 60
				else:
					H = (((Rp - Gp) / delta) + 4) * 60

				if (Cmax == 0):
					S = 0
				else:
					S = delta / Cmax

				V = Cmax

				# if (H < 0):
				# 	H += 360
				#HArray[row][col] = int(H / 360.0 * 255)
				HArray[row][col] = np.uint8(H / 2)
				SArray[row][col] = np.uint8(S * 255)  #from cvtColor() doc, it should be 255*S
				VArray[row][col] = np.uint8(V * 255)  #otherwise the output will be dark
		
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
	resultArray = np.ndarray(shape=(rows, cols, 3), dtype=np.uint8)
	for row in range(0, rows):
		for col in range(0, cols):
			#H = float(HArray[row][col][0]) * 360 / 255 #/ 255 * 360
			#H = float(HArray[row][col][0]) * 2
			H = HArray[row][col][0] * 2
			S = float(SArray[row][col][0]) / 255
			V = float(VArray[row][col][0]) / 255

			#H = int(H)

			C = V * S
			X = C * (1 - abs(int(H / 60) % 2 - 1))
			#X = C * (1 - abs((H / 60) % 2))
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

			resultArray[row][col][0] = np.uint8((Bp + m) * 255)
			resultArray[row][col][1] = np.uint8((Gp + m) * 255)
			resultArray[row][col][2] = np.uint8((Rp + m) * 255)

	return resultArray

def histogramEq(title):
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
		for picName in dirName:
			if (picName.endswith('.jpg')):
				title = picName.split('.')[0]
				convertBGRToHSV(title)
				convertHSVToBGR(title)
				histogramEq(title)

	else:
		print 'No picture folder found in current directory!\n'
	pass