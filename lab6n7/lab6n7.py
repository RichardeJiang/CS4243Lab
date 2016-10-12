import cv2
import numpy as np

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

def quat2rot(q):
	rotationM = []
	row0 = [q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2, 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])]
	row1 = [2*(q[1]*q[2] + q[0]*q[3]), q[0]**2 + q[2]**2 - q[1]**2 - q[3]**2, 2*(q[2]*q[3] - q[0]*q[1])]
	row2 = [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), q[0]**2 + q[3]**2 - q[1]**2 - q[2]**2]
	rotationM = [row0, row1, row2]
	return np.matrix(rotationM)

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

	point2 = quatmult(q, initialPos)
	point2 = quatmult(point2, qp)
	point2[0] = 0
	theta = theta * 2
	q, qp = computeQnQp(theta)

	point3 = quatmult(q, initialPos)
	point3 = quatmult(point3, qp)
	point3[0] = 0
	theta = theta * 1.5
	q, qp = computeQnQp(theta)

	point4 = quatmult(q, initialPos)
	point4 = quatmult(point4, qp)
	point4[0] = 0

	print point2, point3, point4

	### part 1.3 ###
	theta = 0
	q, qp = computeQnQp(theta)
	quatmat_1 = quat2rot(q)
	print quatmat_1

	theta = np.pi / 6.0
	q, qp = computeQnQp(theta)
	quatmat_2 = quat2rot(q)

	theta = theta * 2
	q, qp = computeQnQp(theta)
	quatmat_3 = quat2rot(q)

	theta = theta * 1.5
	q, qp = computeQnQp(theta)
	quatmat_4 = quat2rot(q)
	pass