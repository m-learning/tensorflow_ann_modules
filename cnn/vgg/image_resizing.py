'''
Created on Oct 17, 2016
Image resizing for VGG network
@author: Levan Tsinadze
'''
import cv2

vgg_dim = 224
vgg_size = (vgg_dim, vgg_dim)

class vgg_image_resizer(object):
  """
    Reizes images for VGG network
  """
  
  def resize_image(self, im):
    """
    Resizes image for VGG network
    Args: 
      im image to resize as tensor
    Returns: 
      res_im resized image as tensor
    """
    
    im_h, im_w = im.shape[:2]
    if im_h < vgg_dim or im_w < vgg_dim:
      res_im = cv2.resize(im, vgg_size, interpolation=cv2.INTER_CUBIC)
    elif im_h > vgg_dim or im_w > vgg_dim:
      res_im = cv2.resize(im, vgg_size, interpolation=cv2.INTER_AREA)
    else:
      res_im = im
      
    return res_im
    

  def read_and_resize(self, image_path):
    """
    Resizes image for VGG network
    Args: 
      image_path path to image
    Returns: 
      res_im resized image as tensor
    """
    im = cv2.imread(image_path, 1)
    res_im = self.resize_image(im)
    
    return res_im
  
  def save_resized(self, im, write_path):
    """
    Saves image to specific path
    Args: 
      im image
      write_path path where to save image
    """
    cv2.imwrite(write_path, im)
    
  def read_resize_write(self, image_path, write_path):
    """
    Resizes image for VGG network
    Args: 
      image_path path to image
      write_path path to save resized image
    """
    im = self.read_and_resize(image_path)
    self.save_resized(im, write_path)
    
    