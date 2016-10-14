import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la

def quatmult(q1, q2):
	out = [0, 0, 0, 0]
	out[0] = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
	out[1] = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
	out[2] = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q1[0] + q1[3]*q2[1]
	out[3] = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]
	return out

def computeQnQp(theta):
	normal = [0, 1, 0]
	sinValue = np.sin(theta / 2.0)
	v = [sinValue, sinValue, sinValue]
	vq = np.asarray(v) * np.asarray(normal)
	q = [np.cos(theta / 2.0)] + vq.tolist()
	qp = [np.cos(theta / 2.0)] + [-element for element in vq.tolist()]
	return q, qp

def normalizeVector(q):
	size = np.sqrt(reduce(lambda x, y: x**2 + y**2, q))
	return [ele / size for ele in q]

def quat2rot(q):
	rotationM = []
	row0 = [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])]
	row1 = [2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])]
	row2 = [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]
	rotationM = [row0, row1, row2]
	return np.matrix(rotationM)

def projection(type, sp, tf, cf):
	u0 = 0
	v0 = 0
	bu = 1.0
	bv = 1.0
	f = 1.0
	# sp = np.asarray(sp).reshape(3, 1)
	# tf = np.asarray(tf).reshape(3, 1)
	sp = np.asarray(sp)
	tf = np.asarray(tf)
	if type == 'perspective':
		# ufp = f * np.dot(sp - tf, cf[0].T) * bu / np.dot(sp - tf, cf[2].T) + u0
		# vfp = f * np.dot(sp - tf, cf[1].T) * bu / np.dot(sp - tf, cf[2].T) + v0
		ufp = f * np.dot(sp - tf, cf[:,0]) * bu / np.dot(sp - tf, cf[:,2]) + u0
		vfp = f * np.dot(sp - tf, cf[:,1]) * bu / np.dot(sp - tf, cf[:,2]) + v0
	else:
		# ufp = np.dot(sp - tf, cf[0].T) * bu + u0
		# vfp = np.dot(sp - tf, cf[1].T) * bu + v0
		ufp = np.dot(sp - tf, cf[:,0]) * bu + u0
		vfp = np.dot(sp - tf, cf[:,1]) * bu + v0
	# print np.float(ufp), np.float(vfp)
	return np.float(ufp), np.float(vfp)

def drawProjections(XValueSet, YValueSet, identifier):
	fig = plt.figure()

	for index in range(0, len(XValueSet)):
		sub = fig.add_subplot(2, 2, index+1)
		sub.plot(XValueSet[index], YValueSet[index], 'ro')
	plt.savefig(identifier + '.jpg')
	plt.close()
	return

def computeHomography(pPoints, cPoints):
	print 'pPoint is: ', pPoints
	print 'cPoint is: ', cPoints
	b = [0] * (len(pPoints) * 2)
	M = []
	for index in range(0, len(pPoints)):
		up = np.float(pPoints[index][0])
		vp = np.float(pPoints[index][1])
		uc = np.float(cPoints[index][0])
		vc = np.float(cPoints[index][1])
		M.append([up, vp, 1, 0, 0, 0, -uc*up, -uc*vp, -uc])
		M.append([0, 0, 0, up, vp, 1, -vc*up, -vc*vp, -vc])
	b = np.asmatrix(b).reshape(len(b), 1)
	M = np.asmatrix(M)
	print 'M is: ', M
	np.savetxt('M.txt', M)
	# a, e, r, s = la.lstsq(M, b)
	#a = la.solve(M, b) cannot be used here, as solve can only be used for square matrix M
	u, s, vt = la.svd(M)
	#return a.reshape(3, 3)
	print vt
	print s
	print vt.shape
	#Here we are manually selecting the vector in VT by comparing the S values
	homographyMatrix = vt[-1].reshape(3, 3)
	# print homographyMatrix.shape
	# print homographyMatrix[0]
	# print homographyMatrix[2]
	# print homographyMatrix[2, 2]
	# print 'homo is: ', homographyMatrix[2, 2]
	normalizeValue = 1.0 / homographyMatrix[2, 2]
	return homographyMatrix * normalizeValue

if (__name__ == "__main__"):

	### part 1.1 ###
	pts = np.zeros([11, 3])
	pts[0, :] = [-1, -1, -1]
	pts[1, :] = [1, -1, -1]
	pts[2, :] = [1, 1, -1]
	pts[3, :] = [-1, 1, -1]
	pts[4, :] = [-1, -1, 1]
	pts[5, :] = [1, -1, 1]
	pts[6, :] = [1, 1, 1]
	pts[7, :] = [-1, 1, 1]
	pts[8, :] = [-0.5, -0.5, -1]
	pts[9, :] = [0.5, -0.5, -1]
	pts[10, :] = [0, 0.5, -1]

	### part 1.2 ###
	theta = -np.pi / 6.0
	initialPos = [0, 0, -5]
	q, qp = computeQnQp(theta)
	print q
	initialPos = [0] + initialPos

	frame2 = quatmult(q, initialPos)
	frame2 = quatmult(frame2, qp)
	frame2[0] = 0
	theta = theta * 2
	q, qp = computeQnQp(theta)

	frame3 = quatmult(q, initialPos)
	frame3 = quatmult(frame3, qp)
	frame3[0] = 0
	theta = theta * 1.5
	q, qp = computeQnQp(theta)

	frame4 = quatmult(q, initialPos)
	frame4 = quatmult(frame4, qp)
	frame4[0] = 0

	### part 1.3 ###
	### this part is not implemented correctly according to pdf requirement ###
	quatmat_1 = np.array([(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)])
	quatmat_1 = np.matrix(quatmat_1)
	theta = np.pi / 6.0
	q, qp = computeQnQp(theta)
	#rotationMatrix = quat2rot(normalizeVector(q))
	rotationMatrix = quat2rot(q)

	quatmat_2 = np.dot(rotationMatrix, quatmat_1)
	quatmat_3 = np.dot(rotationMatrix, quatmat_2)
	quatmat_4 = np.dot(rotationMatrix, quatmat_3)


	# theta = 0
	# q, qp = computeQnQp(theta)
	# quatmat_1 = quat2rot(q)
	# print quatmat_1

	# theta = np.pi / 6.0
	# q, qp = computeQnQp(theta)
	# quatmat_2 = quat2rot(q)
	# print quatmat_2

	# theta = theta * 2
	# q, qp = computeQnQp(theta)
	# quatmat_3 = quat2rot(q)
	# print quatmat_3

	# theta = theta * 1.5
	# q, qp = computeQnQp(theta)
	# quatmat_4 = quat2rot(q)
	# print quatmat_4

	### part 2 ###
	perspectivePlotX = []
	perspectivePlotY = []
	orthographicPlotX = []
	orthographicPlotY = []

	#sth worth noting here:
	#in Python, when we append to list in this way:
	#u=sth, and then list.append(u)
	#after that, u=sth else, and again list.append(u)
	#then in the end all elements in list will be the same
	#since 
	for point in pts:
		u = range(0, 4)
		v = range(0, 4)
		u[0], v[0] = projection('perspective', point, initialPos[1:], quatmat_1)
		u[1], v[1] = projection('perspective', point, frame2[1:], quatmat_2)
		u[2], v[2] = projection('perspective', point, frame3[1:], quatmat_3)
		u[3], v[3] = projection('perspective', point, frame4[1:], quatmat_4)
		perspectivePlotX.append(u)
		perspectivePlotY.append(v)

		u[0], v[0] = projection('orthographic', point, initialPos[1:], quatmat_1)
		u[1], v[1] = projection('orthographic', point, frame2[1:], quatmat_2)
		u[2], v[2] = projection('orthographic', point, frame3[1:], quatmat_3)
		u[3], v[3] = projection('orthographic', point, frame4[1:], quatmat_4)
		orthographicPlotX.append(u)
		orthographicPlotY.append(v)

	perspectivePlotX = np.asarray(perspectivePlotX).T.tolist()
	perspectivePlotY = np.asarray(perspectivePlotY).T.tolist()
	orthographicPlotX = np.asarray(orthographicPlotX).T.tolist()
	orthographicPlotY = np.asarray(orthographicPlotY).T.tolist()

	drawProjections(perspectivePlotX, perspectivePlotY, 'perspective')
	drawProjections(orthographicPlotX, orthographicPlotY, 'orthographic')

	### part 3 ###
	paramPointList = []
	paramImagePointList = []
	paramPointList.append(pts[0][:2].tolist())
	paramPointList.append(pts[1][:2].tolist())
	paramPointList.append(pts[2][:2].tolist())
	paramPointList.append(pts[3][:2].tolist())
	paramPointList.append(pts[8][:2].tolist())

	paramImagePointList.append([np.float(perspectivePlotX[2][0]), np.float(perspectivePlotY[2][0])])
	paramImagePointList.append([np.float(perspectivePlotX[2][1]), np.float(perspectivePlotY[2][1])])
	paramImagePointList.append([np.float(perspectivePlotX[2][2]), np.float(perspectivePlotY[2][2])])
	paramImagePointList.append([np.float(perspectivePlotX[2][3]), np.float(perspectivePlotY[2][3])])
	paramImagePointList.append([np.float(perspectivePlotX[2][8]), np.float(perspectivePlotY[2][8])])

	homographyMatrix = computeHomography(paramPointList, paramImagePointList)
	print homographyMatrix

	pass