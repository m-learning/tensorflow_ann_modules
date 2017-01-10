"""
Created on Jan 10, 2017

Saves detected faces in file

@author: Levan Tsinadze
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from skimage import io
import sys

import cv2
import dlib


# Take the image file name from the command line
file_name = sys.argv[1]
out_image = sys.argv[2]

# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()

# Load the image into an array
image = io.imread(file_name)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)

print("I found {} faces in the file {}".format(len(detected_faces), file_name))


# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

  # Detected faces are returned as an object with the coordinates 
  # of the top, left, right and bottom edges
  print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

  # Draw a box around each face we found
  cv2.rectangle(image, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), (255, 0, 255), 2)
  
  (left, top, right, bottom) = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
  print(left, top, right, bottom)
  roi = image[left:top, right:bottom]
  path = os.path.join(out_image, 'aligned_face_{}_crp.jpg'.format(i))
  print(path)
  cv2.imwrite(path.format(i), roi)
  
# Write in new file
path = os.path.join(out_image, 'aligned_face_final.jpg')
cv2.imwrite(path, image)
          
# Wait until the user hits <enter> to close the window          
dlib.hit_enter_to_continue()
