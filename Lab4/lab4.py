import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def gauss_kernels(size, sigma=1.0):
	## return a 2d gaussian gauss_kernels
	if size<3:
		size = 3
	m = size/2
	x, y=np.mgrid[-m:m+1, -m:m+1]
	kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
	kernel_sum = kernel.sum()
	if not sum == 0:
		kernel = kernel / kernel_sum

	return kernel

def convolve(image, ff):
	result = np.zeros(image.shape)
	ffConvolution = np.flipud(ff)
	ffConvolution = np.fliplr(ffConvolution)

	rows = len(image)
	cols = len(image[0])

	for row in range(1, rows-1):
		for col in range(1, cols-1):
			currWindow = image[row-1:row+2, col-1:col+2]
			#strength = np.int(np.sum(currWindow * ffConvolution))
			strength = np.sum(currWindow * ffConvolution)
			result[row][col] = strength

	# result = np.sqrt(result)
	# maxValue = np.amax(result)
	# result *= (255.0 / np.float(maxValue))
	# result = result.astype('uint8')

	return result

if (__name__ == "__main__"):
	SobelKernelV = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)])
	SobelKernelH = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
	k = 0.06

	fileList = os.listdir('.')
	for fileName in fileList:
		if fileName.endswith('.jpg'):
			image = cv2.imread(fileName, 0)
			gx = convolve(image, SobelKernelH)
			gy = convolve(image, SobelKernelV)

			print gx.max()

			I_xx = gx * gx
			I_xy = gx * gy
			I_yy = gy * gy
			
			Gauss_kernel = gauss_kernels(3)

			print I_xx.max()

			W_xx = convolve(I_xx, Gauss_kernel)
			W_xy = convolve(I_xy, Gauss_kernel)
			W_yy = convolve(I_yy, Gauss_kernel)

			print W_xx.max()

			rows = len(W_xx)
			cols = len(W_xx[0])
			responses = np.zeros(W_xx.shape)
			for row in range(10, rows, 10):
				for col in range(10, cols, 10):
					if (row == (rows - 1)) or (col == (cols - 1)):
						pass
					else:
						W = np.zeros((2, 2))
						W[0][0] = W_xx[row][col]
						W[0][1] = W_xy[row][col]
						W[1][0] = W_xy[row][col]
						W[1][1] = W_yy[row][col]
						W = np.asmatrix(W)
						detW = np.linalg.det(W)
						#print detW
						traceW = np.trace(W)
						#print traceW
						response = detW - k * traceW * traceW
						responses[row][col] = response

			#maxResponse = np.amax(responses)
			maxResponse = responses.max()
			print maxResponse
			threshold = np.float(maxResponse * 0.90)
			print threshold
			qualifiedX = []
			qualifiedY = []
			for row in range(10, rows, 10):
				for col in range(10, cols, 10):
					if (row == (rows - 1)) or (col == (cols - 1)):
						pass
					else:
						if (responses[row][col] >= threshold):
							qualifiedX.append(row)
							qualifiedY.append(col)
			qualifiedX = np.asarray(qualifiedX)
			qualifiedY = np.asarray(qualifiedY)

			print qualifiedX
			print qualifiedY

			fig = plt.figure()
			plt.imshow(image, cmap = 'gray')
			plt.hold(True)
			plt.scatter(qualifiedX, qualifiedY, s = 9, marker = 's', color='blue', alpha = .4)
			plt.show()

			#plt.close()
			break