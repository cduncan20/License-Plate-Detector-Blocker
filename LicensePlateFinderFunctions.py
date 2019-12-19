# Author: Casey Duncan
# Date: 12/18/2019
# Description: The script below is used to detect & mask a Colorado license plates (LPs) within an image. Additionally, it estimates the position & coordinate plane
# of the vehicle with respect to the camera taking the photo.

import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib
import matplotlib.pyplot as plt
import urllib
import imutils
import sys

# Function LicensePlateFinder(img) finds the Colorado License Plate (LP) in the inputed image and outputs the pixel locations of the four corners of the LP.
# Input: img - Image or video frame of the vehicle with the Colorado LP shown.
# Output: corner_points - pixel locations of the four corners of the detected LP.
def LicensePlateFinder(img):
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) # Convert to gray
	gray = cv2.bilateralFilter(gray, 5, 100, 100)

	edged = cv2.Canny(gray, 100, 200) #Perform Edge detection

	# Find Characters (contours)
	contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	contours = sorted(contours, key = cv2.contourArea, reverse = True)

	# loop over our contours
	p_plates = []
	for i in range(0,len(contours)):
		c = contours[i]
		min_y = 100000
		min_x = 100000
		max_x = 0
		max_y = 0
		for j in range(0,len(c)):
			point = c[j]
			x = point[0][0]
			y = point[0][1]
			if y < min_y:
				min_y = y
			if x < min_x:
				min_x = x
			if y > max_y:
				max_y = y
			if x > max_x:
				max_x = x
		if max_y - min_y > 0.1:
			box_wl_ratio = (max_x - min_x)/(max_y - min_y)
			box_area = (max_x - min_x)*(max_y - min_y)
		t_range = 0.10
		thresh_max = 0.5 + t_range
		thresh_min = 0.5 - t_range
		# cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,0,255),2)
		# cv2.drawContours(img, c, -1, (0,255,0), 2)
		if box_wl_ratio >= thresh_min and box_wl_ratio <= thresh_max:
			# output_string = f"Contour {i}: Width / Length = {box_wl_ratio}, Dims = {[box_area,min_y,min_x,max_y,max_x]} \n"
			# sys.stdout.write(output_string)
			# sys.stdout.flush()
			p_plates.append([box_area,min_y,min_x,max_y,max_x])
			# cv2.drawContours(img, c, -1, (0,255,0), 2)
			# cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,0,255),2)
	# cv2.imshow('image', img)
	# cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

	p_plates = sorted(p_plates, reverse = True)
	p_nums = []
	nums_counter = 0
	for p_i in p_plates:
		counter = 0
		pi_area = p_i[0]
		pi_min_y = p_i[1]
		pi_min_x = p_i[2]
		
		pi_max_y = p_i[3]
		pi_max_x = p_i[4]
		pi_width = pi_max_x - pi_min_x
		pi_height = pi_max_y - pi_min_y

		for p_j in p_plates:
			pj_area = p_j[0]
			pj_min_y = p_j[1]
			pj_min_x = p_j[2]
			pj_max_y = p_j[3]
			pj_max_x = p_j[4]
			pj_width = pj_max_x - pj_min_x
			pj_height = pj_max_y - pj_min_y

			# if pi_height < pj_height+5 and pi_height > pj_height-5 and pi_area > pj_area-25 and pi_area < pj_area+25:
			# 	counter += 1

			# Check if bounding box min_y points are close
			n = 1.5
			n_a = 3
			if pi_min_y < pj_min_y+pj_height/n and pi_min_y > pj_min_y-pj_height/n:
				# Check if bounding box max_y points are close
				if pi_max_y < pj_max_y+pj_height/n and pi_max_y > pj_max_y-pj_height/n:
					# Check if bounding box widths are close
					if pi_width < pi_width+pi_width/n and pi_width > pi_width-pi_width/n:
						# Check if bounding box areas are close
						if pi_area < pj_area+pj_area/n_a and pi_area > pj_area-pj_area/n_a:
							counter += 1

		for p_j in p_plates:
			pj_area = p_j[0]
			pj_min_y = p_j[1]
			pj_min_x = p_j[2]
			pj_max_y = p_j[3]
			pj_max_x = p_j[4]
			pj_width = pj_max_x - pj_min_x
			pj_height = pj_max_y - pj_min_y

			if pi_width > 0.1:
				width = pj_max_x - pi_min_x
				width_ratio = width / pi_width

			if pi_height > 0.1:
				length = pj_max_y - pi_min_y
				length_ratio = length / pi_height

			if width_ratio > 7 and width_ratio < 10 and length_ratio > 0.5 and length_ratio < 2.5:
				if pi_min_y < pj_min_y+pj_height/n and pi_min_y > pj_min_y-pj_height/n:
					# Check if bounding box max_y points are close
					if pi_max_y < pj_max_y+pj_height/n and pi_max_y > pj_max_y-pj_height/n:
						# Check if bounding box widths are close
						if pi_width < pi_width+pi_width/n and pi_width > pi_width-pi_width/n:
							# Check if bounding box areas are close
							if pi_area < pj_area+pj_area/n_a and pi_area > pj_area-pj_area/n_a:			
								if counter > 5:
									area = (pj_max_x - pi_min_x)*(pj_max_y - pi_min_y)
									if nums_counter == 0:
										# output_string = f"Contour p_i: {p_i} and Contour p_j: {p_j}. Area = {area}, Counter = {counter} \n"
										# sys.stdout.write(output_string)
										# sys.stdout.flush()

										p_nums.append([area, pi_min_x, pi_min_y, pj_max_x, pj_max_y, pj_height])
										# cv2.rectangle(img,(pi_min_x,pi_min_y),(pj_max_x,pj_max_y),(0,0,255),2)
										# cv2.imshow('image', img)
										# cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
										nums_counter += 1
									elif [area, pi_min_x, pi_min_y, pj_max_x, pj_max_y, pj_height] != p_nums[nums_counter-1]:
										# output_string = f"Contour p_i: {p_i} and Contour p_j: {p_j}. Area = {area}, Counter = {counter} \n"
										# sys.stdout.write(output_string)
										# sys.stdout.flush()

										p_nums.append([area, pi_min_x, pi_min_y, pj_max_x, pj_max_y, pj_height])
										# cv2.rectangle(img,(pi_min_x,pi_min_y),(pj_max_x,pj_max_y),(0,0,255),2)
										# cv2.imshow('image', img)
										# cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
										nums_counter += 1
	if len(p_nums) > 0:
		max_area = 0
		for p in p_nums:
			p_area = p[0]
			if p_area > max_area:
				max_area = p_area
				num_bb = [p[1], p[2], p[3], p[4], p[5]]

		cv2.rectangle(img,(num_bb[0],num_bb[1]),(num_bb[2],num_bb[3]),(0,0,255),2)

		c1 = [num_bb[0], num_bb[1]]
		c2 = [num_bb[2], num_bb[1]]
		c3 = [num_bb[0], num_bb[3]]
		c4 = [num_bb[2], num_bb[3]]
		corner_points = np.float32([c1, c2, c3, c4])

		output_string = f"License Plate Points: {corner_points} \n"
		sys.stdout.write(output_string)
		sys.stdout.flush()
	else:
		c1 = [0, 0]
		c2 = [0, 0]
		c3 = [0, 0]
		c4 = [0, 0]
		corner_points = np.float32([c1, c2, c3, c4])
	return corner_points

# Function MapCover(P_car, cover_img, car_img) masks the license plate (LP) on a vehicle by placing a mapped photo over the detected LP.
# Input: P_car - pixel locations of the four corners of the detected LP
#		 cover_img - Photo of image that will be masking the LP
#		 car_img - Photo of vehicle with LP shown
# Output: car_img - Photo of vehicle with LP covered
def MapCover(P_car, cover_img, car_img):
	cover_size = cover_img.shape
	c1 = [0,0]
	c2 = [cover_size[1], 0]
	c3 = [0, cover_size[0]]
	c4 = [cover_size[1], cover_size[0]]
	P_cover = np.float32([c1, c2, c3, c4])

	car_size = car_img.shape
	x_max = car_size[1]
	y_max = car_size[0]
	H = cv2.getPerspectiveTransform(P_cover,P_car)
	mapped_cover_img = cv2.warpPerspective(cover_img,H,(x_max, y_max))

	mapped_cover_img_size = mapped_cover_img.shape
	for n in range(0,mapped_cover_img_size[2]):
		for i in range(0,mapped_cover_img_size[0]):
			for j in range(0,mapped_cover_img_size[1]):
				if mapped_cover_img[i][j][n] > 0:
					car_img[i][j][n] = mapped_cover_img[i][j][n]

	return car_img

# Function DrawCarCoordinates(License_Corners, car_img) estimates the pose of the Vehicle License Plate (LP) with respect to the camera
# Input: License_Corners - pixel locations of the four corners of the detected LP
#		 car_img - Photo of vehicle with license plate shown
# Output: car_img - Photo of vehicle with the vechile's coordinate plane drawn onto the LP
def DrawCarCoordinates(License_Corners, car_img):
	#
	LPD1 = [-10.25/2, 2.5/2, 0]
	LPD2 = [10.25/2, 2.5/2, 0]
	LPD3 = [-10.25/2, -2.5/2, 0]
	LPD4 = [10.25/2, -2.5/2, 0]
	LicensePlateDims = np.float32([LPD1, LPD2, LPD3, LPD4])
	License_Corners = np.float32(License_Corners)

	f = 3058.8
	cx = 4032/2
	cy = 3024/2
	K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype = "double")
	dist = np.zeros((4,1)) # Assuming no lens distortion
	(success, rotVec, transVec) = cv2.solvePnP(LicensePlateDims, License_Corners, K, dist, flags=cv2.cv2.SOLVEPNP_ITERATIVE)

	axis_len = 5.0
	org_x = 0.0
	org_y = 0.0
	org_z = 0.0
	origin = (org_x,org_y,org_z)
	x_axis = (axis_len+org_x,org_y,org_z)
	y_axis = (org_x,axis_len+org_y,org_z)
	z_axis = (org_x,org_y,axis_len+org_z)
	pointsAxes = np.array([origin, x_axis, y_axis, z_axis])

	(pointsImage, jacobian) = cv2.projectPoints(pointsAxes, rotVec, transVec, K, dist)
	pointsImage = np.reshape(pointsImage, (4,2))
	pointsImage = pointsImage.astype(int)
	print(pointsImage)
	origin = (pointsImage[0][0], pointsImage[0][1])
	x_axis = (pointsImage[1][0], pointsImage[1][1])
	y_axis = (pointsImage[2][0], pointsImage[2][1])
	z_axis = (pointsImage[3][0], pointsImage[3][1])

	cv2.line(car_img, origin, x_axis, (0, 0, 255), 2)
	cv2.line(car_img, origin, y_axis, (0, 255, 0), 2) 
	cv2.line(car_img, origin, z_axis, (255, 0, 0), 2) 

	return car_img