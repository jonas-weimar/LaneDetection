#
# Car Extension System - Lane Detection
#
# Version: 0.0.2
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
}

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
detecionLineYCoordinate = int((steeringAdvisorySectionLine + int(normalizedFrameSize[1] * .6)) / 2)

#
# The lane detection takes place in a trapezeoid shaped area as
# a bounding box. This area is being represented via the following array
# of points (x, y).
#

polygonStencil = np.array([
	[int(normalizedFrameSize[0] * .18), steeringAdvisorySectionLine],
	[int(normalizedFrameSize[0] * .42), int(normalizedFrameSize[1] * .6)],
	[int(normalizedFrameSize[0] * .58), int(normalizedFrameSize[1] * .6)],
	[int(normalizedFrameSize[0] * .82), steeringAdvisorySectionLine]
])

showPolygonStencil = True
polygonStencilAsInt32 = np.int32([polygonStencil])

#
# This funtion returns an x value for every line that
# goes through the given points at the given y value.
# Mathematically 'y = m(x - x1) + y1' solved for x
#

def determineXFromPointsAndY(pts, y):
	x1, y1, x2, y2 = pts
	lineGradient = (y2 - y1) / (x2 - x1)
	return ((y - y1) + (lineGradient * x1)) / lineGradient

#
# As long as the camera connection is opened, every second frame is
# getting picked, grayscaled and further treated and analysed.
#

frameCounter = 0
while(cap.isOpened()):
	if frameCounter % 2 == 0:
		frameCounter = frameCounter + 1
		continue

	retval, frame = cap.read()

	if retval != True:
		break

	frame = cv2.resize(frame, normalizedFrameSize)
	grayScaledFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
		# This big loop has three main tasks:
		# * Filter for lines with an absolute angle between 15 and 85 degrees
		# 	to enhance lane detection accuracy.
		# * Draw all chosen lines onto the frame
		# * Gather all x values of the lines left and right of the car
		#   at the given detection y coordinate
		#

		leftXValues = []
		rightXValues = []
		for line in houghLinesProbabilityResult:
			x1, y1, x2, y2 = line[0]
			angle = (np.arctan2(y2 - y1, x2 - x1) * 180. / np.pi)
			if abs(angle) > 15 and abs(angle) < 85:
				# Draw the line
				cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
				# Gather x values for detection line coordinate
				if x2 < bisectionLineYAxis:
					leftXValues.append(
						determineXFromPointsAndY(line[0], detecionLineYCoordinate)
					)
				else:
					rightXValues.append(
						determineXFromPointsAndY(line[0], detecionLineYCoordinate)
					)

		#
		# To draw our indication lines for lane middle and lane averages,
		# we first need to calculate their x values from the gathered average
		# x information.
		#

		leftXAverage = int(sum(leftXValues) / len(leftXValues))
		rightXAverage = int(sum(rightXValues) / len(rightXValues))
		approximateLaneMiddle = int((leftXAverage + rightXAverage) / 2)

		#
		# To indicate the car middle, the lane middle and the distance
		# between them to the user, we need 3 more lines. These lines are
		# calculated using the approx. lane middle and other predefined values
		# and stored in 3 array's of tuples.
		#

		carPositionIndicationLine = [
			(bisectionLineYAxis, detecionLineYCoordinate),
			(bisectionLineYAxis, steeringAdvisorySectionLine)
		]
		carLaneDistanceLine = [
			(bisectionLineYAxis, detecionLineYCoordinate),
			(approximateLaneMiddle, detecionLineYCoordinate)
		]
		laneCenterIndicationLine = [
			(approximateLaneMiddle, carLaneDistanceLine[0][1] - 15),
			(approximateLaneMiddle, carLaneDistanceLine[0][1] + 15)
		]
		leftIndicationLine = [
			(leftXAverage, carLaneDistanceLine[0][1] - 25),
			(leftXAverage, carLaneDistanceLine[0][1] + 25)
		]
		rightIndicationLine = [
			(rightXAverage, carLaneDistanceLine[0][1] - 25),
			(rightXAverage, carLaneDistanceLine[0][1] + 25)
		]

		#
		# Drawing the lines previously calculated on the screen using
		# the previously mentioned line method.
		#

		cv2.line(frame, carLaneDistanceLine[0], carLaneDistanceLine[1], (255, 255, 255), thickness=2)
		cv2.line(frame, carPositionIndicationLine[0], carPositionIndicationLine[1], (0, 255, 255), thickness=2)
		cv2.line(frame, laneCenterIndicationLine[0], laneCenterIndicationLine[1], (0, 255, 0), thickness=2)
		cv2.line(frame, leftIndicationLine[0], leftIndicationLine[1], (0, 255, 255), thickness=2)
		cv2.line(frame, rightIndicationLine[0], rightIndicationLine[1], (0, 255, 255), thickness=2)

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
	except ZeroDivisionError as e:
		print(e)
		pass

	#
	# When s gets pressed, it toggles the 'showPolygonStencil' variable.
	# This way we can toggle the polygonStencil's visibility.
	#

	showPolygonStencil = (cv2.waitKey(1)&0xFF == ord('s')) == (showPolygonStencil ^ True)
	if showPolygonStencil:
		cv2.polylines(frame, polygonStencilAsInt32, True, (0,255,0), 2)

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