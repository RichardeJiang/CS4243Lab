import numpy as np
import numpy.linalg as la
import os

def readFile(filePath):
	fp = open(filePath, 'r')
	data = np.genfromtxt(fp, delimiter=',')
	return data

def genMnB(matrix):
	M = matrix[:,2:4]
	b = matrix[:,0:2]
	b = np.reshape(b, (len(b) * 2, 1))
	M = M.tolist()
	newM = []
	for row in M:
		topRow = row + [1,0,0,0]
		newM.append(topRow)
		botRow = [0,0,0] + row +[1]
		newM.append(botRow)
	M = np.asmatrix(newM)
	b = np.matrix(b)
	return M, b

if (__name__=="__main__"):
	fileList = os.listdir('.')
	if any('data.txt' in fileName for fileName in fileList):
		filePath = 'data.txt'
		matrix = readFile(filePath)
		print 'Raw data.txt is: \n', matrix
		M, b = genMnB(matrix)
		print '\nM is : \n', M
		print '\nb is: \n', b
		a, e, r, s = la.lstsq(M, b)
		print '\nSolved a is: \n', a
		sumSquaredError = la.norm(M * a - b) ** 2
		print '\nComputed sum squared error is: \n', sumSquaredError
		print '\ne is: \n', e
	else:
		print 'No data.txt file found in directory!\n'
	pass