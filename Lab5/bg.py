import cv2
import cv2.cv as cv
import numpy as np
import os

if (__name__ == "__main__"):
	if ('traffic.mp4' in os.listdir('.')):
		cap = cv2.VideoCapture('traffic.mp4')
		fps = cap.get(cv.CV_CAP_PROP_FPS)
		frameCount = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)
		frameWidth = cap.get(cv.CV_CAP_PROP_FRAME_WIDTH)
		frameHeight = cap.get(cv.CV_CAP_PROP_FRAME_HEIGHT)

		print "FPS is: ", fps
		print "Frame count is: ", frameCount
		print "Frame width is: ", frameWidth
		print "Frame height is: ", frameHeight

		fps = np.int(fps)
		frameCount = np.int(frameCount)
		frameHeight = np.int(frameHeight)
		frameWidth = np.int(frameWidth)
	else:
		pass
	pass