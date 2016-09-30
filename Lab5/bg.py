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
		print "Frame count is: ", frameCount # not an integer due to fps is not an integer
		print "Frame width is: ", frameWidth
		print "Frame height is: ", frameHeight

		fps = np.int(fps)
		frameCount = np.int(frameCount)
		frameHeight = np.int(frameHeight)
		frameWidth = np.int(frameWidth)

		_, img = cap.read()
		avgImg = np.float32(img)

		for fr in range(1, frameCount):
			alpha = 0.0092
			# effect of alpha: suitable value is around 0.01
			# the idea is: the bigger alpha is, the faster it is to catch up on the current frame
			# in real application, 0.1 will show the flow of the car (still a moving result)
			# while 0.001 will have the initial frame fix until the end

			cv2.accumulateWeighted(img, avgImg, alpha)
			normImg = cv2.convertScaleAbs(avgImg) #convert it to uint8 for display

			cv2.imshow("Img", img)
			cv2.imshow("NormImg", normImg)
			print "fr = ", fr, "alpha = ", alpha
			cv2.waitKey(1) # add this for high gui to process the imshow() request so that human can see
			_, img = cap.read()

		cv2.waitKey(0) # when 0 is input, it waits infinitely for keyboard input
		cv2.destroyAllWindows()
		cap.release()
	else:
		pass
	pass