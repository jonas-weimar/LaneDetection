#
# Car Extension System - Lane Detection
#
# Version: 0.0.1
# Authors:
#	- Jonas Weimar <jonas-weimar@web.de>
#
# This script performs multiple actions on a single camera input, provided
# through the user when starting the script. These actions among other
# things contain: lane detection; steering advisory; and more.
#
# To run: 'python index.py -v 1 -t 25 -m 40 -g 90' or 'python index.py' for standard settings
# For Help run: 'python index.py -h'
#
# Todo:
#		- Polystencil middle cut out
#		- Lane middle calculation is still a bit off
#

import argparse
import numpy as np
import cv2

#
# As this is a cli application we need an interface to
# specify settings. These settings will be saved to a settings object
# to be referenced later on. To give the user the ability to pass on
# settings we use argparse.
#

parser = argparse.ArgumentParser(description="")
parser.add_argument("-v", "--camera", help="Pass the camera id. Let empty for standard (0).", type=int)
parser.add_argument("-t", "--threshold", help="Pass a threshold for the HoughLines Probability function. Best used between 25-50.", type=int)
parser.add_argument("-l", "--lines", help="Output vector of lines for the HoughLines Probability function. Best used 30.", type=int)
parser.add_argument("-m", "--min-line-length", help="Pass a minimum line length in px for the HoughLines Probability function. Best used between 25-40.", type=int)
parser.add_argument("-g", "--max-line-gap", help="Pass a maximum line gap in px for the HoughLines Probability function. Best used between 40-55.", type=int)
parser.add_argument("-c", "--detect-cars", help="If flag set, the program will additionally identify cars via a cascade classifier.", action="store_true")
parser.add_argument("-p", "--detect-persons", help="If flag set. The program will additionally identify persons via a cascade classifier.", action="store_true")
userPassedSettings = parser.parse_args()

#
# Settings dictionary. All user defined or predefined settings will
# be saved in here.
#

programSettings = {
	"cameraIdentificationNumber": userPassedSettings.camera or 0,
	"houghLinesThreshold": userPassedSettings.threshold or 30,
	"houghLinesVector": userPassedSettings.lines or 30,
	"houghLinesMinLength": userPassedSettings.min_line_length or 30,
	"houghLinesMaxGap": userPassedSettings.max_line_gap or 40,
	"detectCars": userPassedSettings.detect_cars or False,
	"detectPersons": userPassedSettings.detect_persons or False
}

#
# This function detects cars based on a cascade classifier
# using a pre trained set saved in car_cascade.xml. The function
# uses a multiscale algorithm on the provided frame to identify
# all cascades that look like cars.
#

def detectCars(frame):
	cars_cascade = cv2.CascadeClassifier('weights/car_cascade.xml')
	cars = cars_cascade.detectMultiScale(frame, 1.07, 3)
	return cars

#
# This function detects persons based on a cascade classifier
# using a pre trained set saved in person_cascade.xml. The function
# uses a multiscale algorithm on the provided frame to identify
# all cascades that look like persons.
#

def detectPersons(frame):
	person_cascade = cv2.CascadeClassifier('weights/person_cascade.xml')
	persons = person_cascade.detectMultiScale(frame, 1.07, 3)
	return persons

#
# Instance of the VideoCapture object. Based on this instance
# the frames are picked on which the caluclations are performed
# on. Note: 1 is EpocCam; 3 is CamTwist
#

cap = cv2.VideoCapture(
	programSettings.get("cameraIdentificationNumber")
)

#
# Size of the frame provided by the camera image. Based
# on this size, line proportions and other things will be
# calculated.
#

normalizedFrameSize = (
	int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
	int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)

#
# Important frame values for reference. Mentioned here are
# the middle lines of the frame, sectioned via X and Y axis.
#

bisectionLineYAxis = int(normalizedFrameSize[0] * .5)
bisectionLineXAxis = int(normalizedFrameSize[1] * .5)
steeringAdvisorySectionLine = int(normalizedFrameSize[1] * .87)

#
# The lane detection takes place in a trapezeoid shaped area as
# a bounding box. This area is being represented via the following array
# of points (x, y).
#

polygonStencil = np.array([
	[int(normalizedFrameSize[0] * .2), steeringAdvisorySectionLine],
	[int(normalizedFrameSize[0] * .45), int(normalizedFrameSize[1] * .7)],
	[int(normalizedFrameSize[0] * .55), int(normalizedFrameSize[1] * .7)],
	[int(normalizedFrameSize[0] * .8), steeringAdvisorySectionLine]
])

showPolygonStencil = False
polygonStencilAsInt32 = np.int32([polygonStencil])

#
# As long as the camera connection is opened, every frame is
# getting picked, grayscaled and further treated and analysed.
#

while(cap.isOpened()):
	retval, frame = cap.read()

	if retval != True:
		break

	frame = cv2.resize(frame, normalizedFrameSize)
	grayScaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	#
	# If the right flags were set when starting the program
	# the detection algorithms wil try and detect the flagged
	# objects on the frame.
	#

	detectedObjects = []

	if programSettings.get("detectCars"):
		detectedObjects.append([ "Car", detectCars(frame), (255, 0, 0) ])

	if programSettings.get("detectPersons"):
		detectedObjects.append([ "Person", detectPersons(frame), (242, 7, 234) ])

	#
	# To detect the lane we merge the frame with a mask. This
	# blacks every pixel of the frame, excluding the ones protected
	# by the mask. By doing this we reduce the area to search for
	# edges. Using these edges we build lines upon them using the
	# Hough Lines Probability algorithm.
	#
	# To prevent the program to fail if no line or edges are getting
	# detected, we wrap all this with a try/except block
	#

	try:
		cannyDetectedEdges = cv2.Canny(grayScaledFrame, 90, 150, L2gradient=True)
		sectionMask = np.zeros_like(cannyDetectedEdges)
		
		cv2.fillConvexPoly(sectionMask, polygonStencil, [255, 255, 255])
		maskedFrame = cv2.bitwise_and(cannyDetectedEdges, sectionMask)
		
		houghLinesProbabilityResult = cv2.HoughLinesP(
			maskedFrame, 1, np.pi/180,
			programSettings.get("houghLinesThreshold"),
			lines=programSettings.get("houghLinesVector"),
			minLineLength=programSettings.get("houghLinesMinLength"),
			maxLineGap=programSettings.get("houghLinesMaxGap")
		)

		#
		# Filter for lines with an absolute angle between 15 and 85 degrees
		# to enhance lane detection accuracy.
		#

		temporaryFilteredLines = []
		for line in houghLinesProbabilityResult:
			x1, y1, x2, y2 = line[0]
			angle = (np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi)
			if abs(angle) > 15 and abs(angle) < 85:
				temporaryFilteredLines.append(line)

		houghLinesProbabilityResult = temporaryFilteredLines
		
		#
		# To calculate the approx. lane middle, we take the average of four
		# X values of our detected lines predicted by the Hough Lines
		# algortihm.
		#

		leftMinX = min(line[0][0] for line in filter(lambda i: i[0][0] < bisectionLineYAxis, houghLinesProbabilityResult))
		rightMinX = min(line[0][0] for line in filter(lambda i: i[0][0] > bisectionLineYAxis, houghLinesProbabilityResult))
		leftMaxX = max(line[0][2] for line in filter(lambda i: i[0][0] < bisectionLineYAxis, houghLinesProbabilityResult))
		rightMaxX = max(line[0][2] for line in filter(lambda i: i[0][0] > bisectionLineYAxis, houghLinesProbabilityResult))
		
		approximateLaneMiddle = int(sum([
			leftMinX, leftMaxX, rightMinX, rightMaxX
		]) / 4)
		
		#
		# Draw the detected lines (lane describing lines) onto the frame
		# using openCV's line method.
		#

		for line in houghLinesProbabilityResult:
			x1, y1, x2, y2 = line[0]
			cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)

		#
		# To indicate the car middle, the lane middle and the distance
		# between them to the user, we need 3 more lines. These lines are
		# calculated using the approx. lane middle and other predefined values
		# and stored in 3 array's of tuples.
		#

		carPositionIndicationLine = [
			(bisectionLineYAxis, int((steeringAdvisorySectionLine + bisectionLineXAxis) / 2)),
			(bisectionLineYAxis, steeringAdvisorySectionLine)
		]

		carLaneDistanceLine = [
			(bisectionLineYAxis, int((steeringAdvisorySectionLine + bisectionLineXAxis) / 2)),
			(approximateLaneMiddle, int((steeringAdvisorySectionLine + bisectionLineXAxis) / 2))
		]

		laneCenterIndicationLine = [
			(approximateLaneMiddle, carLaneDistanceLine[0][1] - 15),
			(approximateLaneMiddle, carLaneDistanceLine[0][1] + 15)
		]

		#
		# Drawing the lines previously calculated on the screen using
		# the previously mentioned line method.
		#

		cv2.line(frame, carLaneDistanceLine[0], carLaneDistanceLine[1], (255, 255, 255), thickness=2)
		cv2.line(frame, carPositionIndicationLine[0], carPositionIndicationLine[1], (0, 255, 255), thickness=2)
		cv2.line(frame, laneCenterIndicationLine[0], laneCenterIndicationLine[1], (0, 255, 0), thickness=2)

		#
		# Additionally to the line indication we draw a steering advisory
		# on the screen either saying 'Turn Right' or 'Turn Left'
		#

		steeringAdvisoryText = 'Turn Right' if (carPositionIndicationLine[0][0] - laneCenterIndicationLine[0][0]) < 0 else 'Turn Left'
		steeringAdvisoryTextSize = cv2.getTextSize(steeringAdvisoryText, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
		cv2.putText(frame, steeringAdvisoryText, (int(bisectionLineYAxis - (steeringAdvisoryTextSize[0][0] / 2)), steeringAdvisorySectionLine + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
	except ValueError as e:
		print(e)
		pass
	except TypeError as e:
		print(e)
		pass

	# polygonCoordinatesAsInt32 = np.int32([polygonStencil])
	# cv2.polylines(frame, polygonCoordinatesAsInt32, True, (254, 247, 8), 1)

	#
	# Draw a rectangle around the objects detected via cascade algorithms onto
	# the frame using the rectangle draw method provide by openCV.
	#

	for objectClass in detectedObjects:
		for (x, y, w, h) in objectClass[1]:
			cv2.putText(frame, objectClass[0], (x, y+h), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
			cv2.rectangle(frame, (x, y), (x+w,y+h), color=objectClass[2], thickness=2)

	#
	# When s gets pressed, it toggles the 'showPolygonStencil' variable.
	# This way we can toggle the polygonStencil's visibility.
	#

	showPolygonStencil = (cv2.waitKey(1)&0xFF == ord('s')) == (showPolygonStencil ^ True)
	if showPolygonStencil:
		cv2.polylines(frame, polygonStencilAsInt32, True, (255,0,0), 1)

	#
	# Show the frame on the screen.
	#

	cv2.imshow('frame', frame)
	if cv2.waitKey(1)&0xFF == ord('q'):
		break

#
# If the loop ends the last thing to do
# is to close the video capture and close all windows.
#

cap.release()
cv2.destroyAllWindows()