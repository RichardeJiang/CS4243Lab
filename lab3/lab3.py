import cv2
import numpy as np
import os

def MyConvolve(image, ff):
	result = np.zeros(image.shape)
	if (ff == 1):
		kernelV = np.array([(-1, 0, 1), (-1, 0, 1), (-1, 0, 1)])
		kernelH = np.array([(1, 1, 1), (0, 0, 0), (-1, -1, -1)])
		#maxValue = np.sqrt((255 * 3) ** 2 * 2)
		# The max value is obtained by 1*255+1*255+1*255 and go through the same process below
		title = 'Prewitt'
	else:
		kernelV = np.array([(-1, 0, 1), (-2, 0, 2), (-1, 0, 1)])
		kernelH = np.array([(1, 2, 1), (0, 0, 0), (-1, -2, -1)])
		#maxValue = np.sqrt((255 * 4) ** 2 * 2)
		title = 'Sobel'
	rows = len(image)
	cols = len(image[0])

	for row in range(1, rows-1):
		for col in range(1, cols-1):
			currWindow = image[row-1:row+2, col-1:col+2]
			strength = np.int(np.sum(currWindow * kernelH) ** 2 + np.sum(currWindow * kernelV) ** 2)
			result[row][col] = strength

	result = np.sqrt(result)
	maxValue = np.amax(result)
	# another way of doing normalization is to use the max value in the array
	result *= (255.0 / np.float(maxValue))
	result = result.astype('uint8')

	return result

def edgeDetection(image, title):
	PrewittResult = MyConvolve(image, 1)
	SobelResult = MyConvolve(image, 2)
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
			# 	pass
			# else:
			# 	image[row][col] = 0
			# if (image[row][col] == np.amax(image[row])) or (image[row][col] == np.amax(image[:, col])):
			# 	result[row][col] = image[row][col]

	cv2.imwrite('thinned_sobel_result_' + title + '.jpg', result)

	return

if (__name__=="__main__"):
	fileList = os.listdir('.')
	# for fileName in fileList:
	# 	if fileName.endswith('.jpg') and ('test' in fileName):
	# 		title = fileName.split('.')[0]
	# 		image = cv2.imread(fileName, 0)
	# 		edgeDetection(image, title)
	# 		edgeThinning(image, title)

	title = 'example'
	image = cv2.imread('example.jpg', 0)
	sobelResult = edgeDetection(image, title)
	edgeThinning(sobelResult, title)
	pass