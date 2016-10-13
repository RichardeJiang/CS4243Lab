import cv2
import numpy as np
import matplotlib.pyplot as plt

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
	sp = np.asarray(sp).reshape(3, 1)
	tf = np.asarray(tf).reshape(3, 1)
	if type == 'perspective':
		ufp = f * np.dot(sp - tf, cf[0]) * bu / np.dot(sp - tf, cf[2]) + u0
		vfp = f * np.dot(sp - tf, cf[1]) * bu / np.dot(sp - tf, cf[2]) + v0
	else:
		ufp = f * np.dot(sp - tf, cf[0]) * bu + u0
		vfp = f * np.dot(sp - tf, cf[1]) * bu + v0
	return ufp, vfp

def drawProjections():
	return

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

	print frame2, frame3, frame4

	### part 1.3 ###
	### this part is not implemented correctly according to pdf requirement ###
	quatmat_1 = np.array([(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)])
	quatmat_1 = np.matrix(quatmat_1)
	theta = np.pi / 6.0
	q, qp = computeQnQp(theta)
	print quatmat_1
	rotationMatrix = quat2rot(normalizeVector(q))

	quatmat_2 = rotationMatrix * quatmat_1
	quatmat_3 = rotationMatrix * quatmat_2
	quatmat_4 = rotationMatrix * quatmat_3


	# theta = 0
	# q, qp = computeQnQp(theta)
	# quatmat_1 = quat2rot(normalizeVector(q))
	# print quatmat_1

	# theta = np.pi / 6.0
	# q, qp = computeQnQp(theta)
	# quatmat_2 = quat2rot(normalizeVector(q))

	# theta = theta * 2
	# q, qp = computeQnQp(theta)
	# quatmat_3 = quat2rot(normalizeVector(q))

	# theta = theta * 1.5
	# q, qp = computeQnQp(theta)
	# quatmat_4 = quat2rot(normalizeVector(q))

	### part 2 ###
	for point in pts:
		projection('perspective', point, initialPos[1:], quatmat_1)
		projection('perspective', point, frame2[1:], quatmat_2)
		projection('perspective', point, frame3[1:], quatmat_3)
		projection('perspective', point, frame4[1:], quatmat_4)

		projection('orthographic', point, initialPos[1:], quatmat_1)
		projection('orthographic', point, frame2[1:], quatmat_2)
		projection('orthographic', point, frame3[1:], quatmat_3)
		projection('orthographic', point, frame4[1:], quatmat_4)
	pass