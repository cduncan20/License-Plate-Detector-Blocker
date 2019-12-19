# Author: Casey Duncan
# Date: 12/18/2019
# Description: The script below is used to detect & mask a Colorado license plates (LPs) within a video. Additionally, it estimates the position & coordinate plane
# of the vehicle with respect to the camera taking the photo.

import cv2
from LicensePlateFinderFunctions import LicensePlateFinder
from LicensePlateFinderFunctions import MapCover
from LicensePlateFinderFunctions import DrawCarCoordinates

# Import Video with License Plate
path_car = r'C:\Users\cdunc\Documents\CSM Grad School Work\2019\Fall\CSCI 507B - Computer Vision\Final Project\Images\IMG_9014.MOV'
cap = cv2.VideoCapture(path_car)

# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = 1024
frame_height = 576

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# Import License Plate Censoring Photo
path_cover = r'C:\Users\cdunc\Documents\CSM Grad School Work\2019\Fall\CSCI 507B - Computer Vision\Final Project\Images\censored.png'
cover_img = cv2.imread(path_cover)
frames_found=0
frames = 0
while(True):
	ret, car_img = cap.read()

	if ret == True:
		scale_factor = 1024 / car_img.shape[1]
		car_img = cv2.resize(car_img, (1024,int(scale_factor*car_img.shape[0]))) # Resize to be a reasonable size

		# Find Corner Points:
		# This function only is able to find the maximum & minimum x,y points for the license plate character contours.
		# Therefore, these are not the true four corners of the license plate or license plate characters.
		License_Corners = LicensePlateFinder(car_img)

		if License_Corners[0][0] > 0:
			# # Map License Plate Cover ontop of license plate corners to cover license plate characters
			car_img = MapCover(License_Corners, cover_img, car_img)
			car_img = DrawCarCoordinates(License_Corners, car_img)
			frames_found = frames_found+1
		frames = frames+1
		
		# Write the frame into the file 'output.avi'
		out.write(car_img)

		# Display the resulting frame
		cv2.imshow('image', car_img)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# Break the loop
	else:
		break 

# When everything done, release the video capture and video write objects
cap.release()
out.release()
 
# Closes all the frames
cv2.destroyAllWindows() 
print("Frames = ", frames)
print("Frames Found = ", frames_found)