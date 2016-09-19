import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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

	return result

if (__name__ == "__main__"):
	SobelKernelH = np.array([(-1, -2, -1), (0, 0, 0), (1, 2, 1)])
	SobelKernelV = np.array([(1, 0, -1), (2, 0, -2), (1, 0, -1)])
	k = 0.06

	fileList = os.listdir('.')
	for fileName in fileList:
		if fileName.endswith('.jpg') and 'corners' not in fileName:
			title = fileName.split('.')[0]
			image = cv2.imread(fileName, 0)

			rows = len(image)
			cols = len(image[0])
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
			print W_xy.max()
			print W_yy.max()

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
						traceW = np.trace(W)
						response = detW - k * traceW * traceW
						responses[row][col] = response

			#maxResponse = np.amax(responses)
			maxResponse = responses.max()
			print maxResponse
			threshold = np.float(maxResponse * 0.8)
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
			plt.scatter(qualifiedY, qualifiedX, s = 81, marker = 's', color='blue', alpha = .4)

			plt.savefig(title + '_corners.jpg')
			#plt.show()
			plt.close()
			
			#break