import cv2
import os
import numpy as np

def convertBGRToHSV(title):
	try:
		image = cv2.imread('lab2_pictures/' + title + '.jpg')
	except:
		print 'Target files not found in current directory!\n'
	else:

		#print len(image) #vertical pixels
		#print len(image[0]) #horizontal pixels
		#print len(image[0][0]) #BGR
		rows = len(image)
		cols = len(image[0])

		HArray = np.zeros(shape=(rows, cols))
		SArray = np.zeros(shape=(rows, cols))
		VArray = np.zeros(shape=(rows, cols))

		for row in range(0, rows):
			for col in range(0, cols):
				B = image[row][col][0]
				G = image[row][col][1]
				R = image[row][col][2]
				Bp = np.float(B) / 255.0
				Gp = np.float(G) / 255.0
				Rp = np.float(R) / 255.0
				Cmax = max (Rp, Gp, Bp)
				Cmin = min (Rp, Gp, Bp)
				delta = np.float(Cmax - Cmin)

				if (delta == 0):
					H = 0
				elif (Cmax == Rp):
					H = (((Gp - Bp) / delta) % 6) * 60
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
				HArray[row][col] = np.round(H / 360.0 * 255).astype('uint8')
				#HArray[row][col] = np.uint8(H / 2)
				SArray[row][col] = np.round(S * 255).astype('uint8')  #from cvtColor() doc, it should be 255*S
				VArray[row][col] = np.round(V * 255).astype('uint8')  #otherwise the output will be dark
		
		cv2.imwrite('output/' + title + '_hue.jpg', HArray)
		cv2.imwrite('output/' + title + '_saturation.jpg', SArray)
		cv2.imwrite('output/' + title + '_brightness.jpg', VArray)
	return

def convertHSVToBGR(title):
	try:
		HArray = cv2.imread('output/' + title + '_hue.jpg', 0)
		SArray = cv2.imread('output/' + title + '_saturation.jpg', 0)
		VArray = cv2.imread('output/' + title + '_brightness.jpg', 0)
	except:
		print 'Target files not found in current directory!\n'
	else:
		resultArray = convertHSVToBGRCalculation(HArray, SArray, VArray)
		cv2.imwrite('output/' + title + '_hsv2rgb.jpg', resultArray)
	return

def convertHSVToBGRCalculation(HArray, SArray, VArray):
	rows = len(HArray)
	cols = len(HArray[0])
	resultArray = np.zeros(shape=(rows, cols, 3))
	for row in range(0, rows):
		for col in range(0, cols):
			#H = float(HArray[row][col][0]) * 2
			H = np.float(HArray[row][col]) / 255.0 * 360
			S = np.float(SArray[row][col]) / 255.0
			V = np.float(VArray[row][col]) / 255.0


			C = V * S
			X = C * (1 - abs((H / 60.0) % 2 - 1))
			m = V - C

			H = np.int(H)

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

			resultArray[row][col][0] = np.round((Bp + m) * 255).astype('uint8')
			resultArray[row][col][1] = np.round((Gp + m) * 255).astype('uint8')
			resultArray[row][col][2] = np.round((Rp + m) * 255).astype('uint8')

	return resultArray

def histogramEq(title):
	try:
		HArray = cv2.imread('output/' + title + '_hue.jpg', 0)
		SArray = cv2.imread('output/' + title + '_saturation.jpg', 0)
		VArray = cv2.imread('output/' + title + '_brightness.jpg', 0)
	except:
		print 'Target files not found in current directory!\n'
	else:
		histogram = np.zeros(256)
		sumPlot = np.zeros(256)
		rows = len(VArray)
		cols = len(VArray[0])
		newVArray = np.zeros(shape=(rows, cols))
		sumOfHis = 0

		for row in range(0, rows):
			for col in range(0, cols):
				histogram[VArray[row][col]] += 1

		for index in range(0, 256):
			sumOfHis += histogram[index]
			sumPlot[index] = sumOfHis

		unit = float(sumOfHis) / 256.0
		for row in range(0, rows):
			for col in range(0, cols):
				newVArray[row][col] = round(sumPlot[VArray[row][col]] / unit)

		# Specify uint8 at the end (after clip), otherwise in the middle there will be overflow already
		newVArray = np.clip(newVArray, 0, 255).astype('uint8')
		
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