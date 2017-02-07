"""
Created on Feb 6, 2017

Draws rectangles around detected faces in image

@author: Levan Tsinadze
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image, ImageDraw
import random

import numpy as np


BASECOLOR = 'yellow'

def generate_color():
  """Generates randor color
    Returns:
      color code
  """
  
  return random.randint(0, 255)

def draw_box(dr, _rec, _color):
  """Draws box on image"""
  
  (x1, y1, x2, y2) = _rec
  rec_dr = [x1, y1, x2, y2]
  dr.rectangle(rec_dr, outline=_color)
  

def draw_matched_faces(faces):
  """Draw rectangles on similar faces
    Args:
      faces - detected faces
  """
        
  (n_faces_0, n_faces_1, rects_0, rects_1, scores, comps, dr0, dr1) = faces
  drawn_1 = [False] * n_faces_1

  for i in range(n_faces_0):
    color = BASECOLOR
    style = 'base'

    k = np.argmax(scores[i])
    if comps[i, k]:
      color = generate_color()
      style = 'match'
      drawn_1[k] = True
      draw_box(dr1, rects_1[k], color, style, k)

    draw_box(dr0, rects_0[i], color, style, i)

  color = BASECOLOR
  for j in range(n_faces_1):
    if not drawn_1[j]:
      draw_box(dr1, rects_1[j], color, 'base', j)

def draw_rectangles(_image, _rectangles, _save_path):
  """Draws rectangles on images
    Args:
      _image - image path
      _rectangles - rectangles to draw
      _save_path - path to save modified image
  """

  if _rectangles and len(_rectangles) > 0:
    im = Image.open(_image)
    dr = ImageDraw.Draw(im)
    for _rec in _rectangles:
        (x1, y1, x2, y2) = (_rec.left(), _rec.top(), _rec.right(), _rec.bottom())
        rec_dr = [x1, y1, x2, y2]
        dr.rectangle(rec_dr)
    im.save(_save_path)
