# Author: Casey Duncan
# Date: 12/18/2019
# Description: The script below is used to detect & mask a Colorado license plates (LPs) within a photo. Additionally, it estimates the position & coordinate plane
# of the vehicle with respect to the camera taking the photo.

import cv2
from LicensePlateFinderFunctions import LicensePlateFinder
from LicensePlateFinderFunctions import MapCover
from LicensePlateFinderFunctions import DrawCarCoordinates

# Import Car Photo
path_car = r'C:\Users\cdunc\Documents\CSM Grad School Work\2019\Fall\CSCI 507B - Computer Vision\Final Project\Images\IMG_8945.JPG'
car_img = cv2.imread(path_car)
scale_factor = 1024 / car_img.shape[1]
car_img = cv2.resize(car_img, (1024,int(scale_factor*car_img.shape[0]))) # Resize to be a reasonable size

print(car_img.shape)
# Find Corner Points:
# This function only is able to find the maximum & minimum x,y points for the license plate character contours.
# Therefore, these are not the true four corners of the license plate or license plate characters.
License_Corners = LicensePlateFinder(car_img)
cv2.imshow('image', car_img)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

# Import License Plate Censoring Photo
path_cover = r'C:\Users\cdunc\Documents\CSM Grad School Work\2019\Fall\CSCI 507B - Computer Vision\Final Project\Images\censored.png'
cover_img = cv2.imread(path_cover)

# Map License Plate Cover ontop of license plate corners to cover license plate characters
# car_img = MapCover(License_Corners, cover_img, car_img)
cv2.imshow('image', car_img)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)

car_img = DrawCarCoordinates(License_Corners, car_img)
cv2.imshow('image', car_img)
cv2.waitKey(0); cv2.destroyAllWindows(); cv2.waitKey(1)
