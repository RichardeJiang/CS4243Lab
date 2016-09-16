import cv2
import numpy as np
import os

def MyConvolve(image, ff):
	result = np.zeros(image.shape)
	ffConvolution = np.flipud(ff)
	ffConvolution = np.fliplr(ffConvolution)

	rows = len(image)
	cols = len(image[0])

	for row in range(1, rows-1):
		for col in range(1, cols-1):
			currWindow = image[row-1:row+2, col-1:col+2]
			strength = np.int(np.sum(currWindow * ffConvolution) ** 2)
			result[row][col] = strength

	# result = np.sqrt(result)
	# maxValue = np.amax(result)
	# result *= (255.0 / np.float(maxValue))
	# result = result.astype('uint8')

	return result

def edgeDetection(image, title):
	PrewittFilterV = np.array([(1, 0, -1), (1, 0, -1), (1, 0, -1)])
	PrewittFilterH = np.array([(-1, -1, -1), (0, 0, 0), (1, 1, 1)])
	SobelFilterV = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
	SobelFilterH = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)])

	PrewittResult = MyConvolve(image, PrewittFilterH) + MyConvolve(image, PrewittFilterV)
	PrewittResult = np.sqrt(PrewittResult)
	maxValue = np.amax(PrewittResult)
	minValue = np.amin(PrewittResult)
	PrewittResult = (PrewittResult - minValue) / np.float(maxValue - minValue) * 255.0
	PrewittResult = PrewittResult.astype('uint8')

	SobelResult = MyConvolve(image, SobelFilterH) + MyConvolve(image, SobelFilterV)
	SobelResult = np.sqrt(SobelResult)
	maxValue = np.amax(SobelResult)
	minValue = np.amin(SobelResult)
	SobelResult = (SobelResult - minValue) / np.float(maxValue - minValue) * 255.0
	SobelResult = SobelResult.astype('uint8')

	cv2.imwrite('prewitt_result_' + title + '.jpg', PrewittResult)
	cv2.imwrite('sobel_result_' + title + '.jpg', SobelResult)
	return SobelResult

def edgeThinning(image, title):
	result = np.zeros(image.shape)
	rows = len(image)
	cols = len(image[0])
	for row in range(1, rows-1):
		for col in range(1, cols-1):
			currWindow = image[row-1:row+2, col-1:col+2]
			if (currWindow[1][1] == np.amax(currWindow[1])) or (currWindow[1][1] == np.amax(currWindow[:, 1])):
				result[row][col] = currWindow[1][1]
	result = result.astype('uint8')

	cv2.imwrite('thinned_sobel_result_' + title + '.jpg', result)

	return

if (__name__=="__main__"):
	fileList = os.listdir('.')
	for fileName in fileList:
		if fileName.endswith('.jpg') and ('test' in fileName):
			title = fileName.split('.')[0]
			image = cv2.imread(fileName, 0)
			sobelResult = edgeDetection(image, title)
			edgeThinning(sobelResult, title)

	# title = 'example'
	# image = cv2.imread('example.jpg', 0)
	# sobelResult = edgeDetection(image, title)
	# edgeThinningTwo(sobelResult, title)
	pass