# -*- coding: utf-8 -*-
"""
Created on 12/16/ 04:00 2020
Laser Sketch
@author: Tgroller
"""
import numba  # We added these two lines for a 500x speedup
import cv2
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import math
import sys, PIL.Image
import numba.cuda

def minmax(v):
    if v > 255:
        v = 255
    if v < 0:
        v = 0
    return v


def sepia(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # converting to RGB as sepia matrix is for RGB
    res = np.array(res, dtype=np.float64)
    res = cv2.transform(res, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))
    res[np.where(res > 255)] = 255 # clipping values greater than 255 to 255
    res = np.array(res, dtype=np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

    return res


def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v,value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


#@st.cache
#@st.cache(suppress_st_warning=True)

def dithering_gray(inMat, samplingF):
    #https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    #https://www.youtube.com/watch?v=0L2n8Tg2FwI&t=0s&list=WL&index=151
    #input is supposed as color
    # grab the image dimensions
    h = inMat.shape[0]
    w = inMat.shape[1]
    #x, y = cuda.grid(2)
    # loop over the image
    for y in range(0, h-1):
        for x in range(1, w-1):
            # threshold the pixel
            old_p = inMat[y, x]
            new_p = np.round(samplingF * old_p/255.0) * (255/samplingF)
            inMat[y, x] = new_p

            quant_error_p = old_p - new_p

            inMat[y, x+1] = minmax(inMat[y, x+1] + quant_error_p * 7 / 16.0)
            inMat[y+1, x-1] = minmax(inMat[y+1, x-1] + quant_error_p * 3 / 16.0)
            inMat[y+1, x] = minmax(inMat[y+1, x] + quant_error_p * 5 / 16.0)
            inMat[y+1, x+1] = minmax(inMat[y+1, x+1] + quant_error_p * 1 / 16.0)

    return inMat
#@st.cache
#@st.cache(suppress_st_warning=True)
#@numba.jit(fastmath = True, parallel=True)    # We added these two lines for a 500x speedup
@numba.jit    # We added these two lines for a 500x speedup
def dithering_color(inMat, samplingF):
    #https://en.wikipedia.org/wiki/Floyd–Steinberg_dithering
    #https://www.youtube.com/watch?v=0L2n8Tg2FwI&t=0s&list=WL&index=151
    #input is supposed as color
    # grab the image dimensions
    h = inMat.shape[0]
    w = inMat.shape[1]

    # loop over the image
    for y in range(0, h-1):
        for x in range(1, w-1):
            # threshold the pixel
            old_b = inMat[y, x, 0]
            old_g = inMat[y, x, 1]
            old_r = inMat[y, x, 2]

            new_b = np.round(samplingF * old_b/255.0) * (255/samplingF)
            new_g = np.round(samplingF * old_g/255.0) * (255/samplingF)
            new_r = np.round(samplingF * old_r/255.0) * (255/samplingF)


            inMat[y, x, 0] = new_b
            inMat[y, x, 1] = new_g
            inMat[y, x, 2] = new_r

            quant_error_b = old_b - new_b
            quant_error_g = old_g - new_g
            quant_error_r = old_r - new_r


            inMat[y, x+1, 0] = minmax(inMat[y, x+1, 0] + quant_error_b * 7 / 16.0)
            inMat[y, x+1, 1] = minmax(inMat[y, x+1, 1] + quant_error_g * 7 / 16.0)
            inMat[y, x+1, 2] = minmax(inMat[y, x+1, 2] + quant_error_r * 7 / 16.0)

            inMat[y+1, x-1, 0] = minmax(inMat[y+1, x-1, 0] + quant_error_b * 3 / 16.0)
            inMat[y+1, x-1, 1] = minmax(inMat[y+1, x-1, 1] + quant_error_g * 3 / 16.0)
            inMat[y+1, x-1, 2] = minmax(inMat[y+1, x-1, 2] + quant_error_r * 3 / 16.0)


            inMat[y+1, x, 0] = minmax(inMat[y+1, x, 0] + quant_error_b * 5 / 16.0)
            inMat[y+1, x, 1] = minmax(inMat[y+1, x, 1] + quant_error_g * 5 / 16.0)
            inMat[y+1, x, 2] = minmax(inMat[y+1, x, 2] + quant_error_r * 5 / 16.0)


            inMat[y+1, x+1, 0] = minmax(inMat[y+1, x+1, 0] + quant_error_b * 1 / 16.0)
            inMat[y+1, x+1, 1] = minmax(inMat[y+1, x+1, 1] + quant_error_g * 1 / 16.0)
            inMat[y+1, x+1, 2] = minmax(inMat[y+1, x+1, 2] + quant_error_r * 1 / 16.0)




    # return the thresholded image
    return inMat

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image


# Get the pixel from the given image
def get_pixel(image, i, j):
  # Inside image bounds?
  width, height = image.size
  if i > width or j > height:
    return None

  # Get Pixel
  pixel = image.getpixel((i, j))
  return pixel

# Create a Grayscale version of the image
def convert_grayscale(image):
  # Get size
  width, height = image.size

  # Create new Image and a Pixel Map
  new = create_image(width, height)
  pixels = new.load()

  # Transform to grayscale
  for i in range(width):
    for j in range(height):
      # Get Pixel
      pixel = get_pixel(image, i, j)

      # Get R, G, B values (This are int from 0 to 255)
      red =   pixel[0]
      green = pixel[1]
      blue =  pixel[2]

      # Transform to grayscale
      gray = (red * 0.299) + (green * 0.587) + (blue * 0.114)

      # Set Pixel in new image
      pixels[i, j] = (int(gray), int(gray), int(gray))

    # Return new image
    return new

def sketchit (img, sketch):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if sketch == "Pencil Sketch":

        value = st.sidebar.slider('Tune the brightness of your sketch (the higher the value, the brighter your sketch)', 0.0, 300.0, 250.0)
        kernel = st.sidebar.slider('Tune the boldness of the edges of your sketch (the higher the value, the bolder the edges)', 1, 99, 25, step=2)
        gray_blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
        sketch = cv2.divide(gray, gray_blur, scale=value)



    if sketch == 'Enhance':
       sigma = st.sidebar.slider('Tune the enahance level of the Sigma Value', 0, 0, 200, step=1)
       delta = st.sidebar.slider('Tune the enhance of the image Delta Value', float(0.0), float(0.07),float(1.0), step =float(.01))
       invert = cv2.detailEnhance(img, sigma_s=sigma, sigma_r=delta)
       sketch = invert

    if sketch == 'Negative':
        invert = cv2.bitwise_not(gray)
        sketch = invert


    if sketch == 'hdr':
        im = cv2.imread(img, cv2.IMREAD_ANYDEPTH)

        tonemapDurand = cv2.createTonemapDurand(2.2)
        ldrDurand = tonemapDurand.process(im)

        im2_8bit = np.clip(ldrDurand * 255, 0, 255).astype('uint8')
        sketch = im2_8bit

    if sketch == 'B/W':
       sketch = gray

    if sketch == 'Sepia':

       image_file = sepia(img.copy())
       sketch = image_file

    if sketch == 'Colorize':
       value = st.sidebar.slider('Change Color', float(0.0), float(300),float(150), step =float(1))
       image_file = change_brightness(img.copy(), value)
       sketch = image_file

    if sketch == 'Color Dither':

       image_file = dithering_color(img.copy(), 1)
       sketch = image_file

    if sketch == 'Grey Dither':
       image_file = dithering_gray(gray.copy(), 1)
       sketch = image_file

    if sketch == 'Negative Dither':
       image_file = dithering_gray(gray.copy(), 1)
       img_invert=cv2.bitwise_not(image_file)
       sketch = img_invert



    if sketch == "WaterColor":
       sigma = st.sidebar.slider('Tune the Sigma Value',0, 200, 5, step=2)
       delta = st.sidebar.slider('Tune the Delta Value', float(0.0), float(0.07),float(1.0), step =float(.01))
       sketch = cv2.stylization(img, sigma_s=sigma, sigma_r=delta)


    return sketch



st.write("""
          # Sketch Your Image!

          """
          )

st.write("This is an app to turn your photos into a Sketch and more - Enjoy Tracy")

file = st.sidebar.file_uploader("Please upload an image file", type=["jpeg", "jpg", "png"])

if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    img = np.array(image)

    option = st.sidebar.selectbox(
    'Which Sketch filters would you like to apply?',
    ('Pencil Sketch',  'Enhance', 'B/W', 'Negative','Sepia', 'Colorize', 'Color Dither', 'Grey Dither', 'WaterColor'))

    st.text("Your original image")
    st.image(image, use_column_width=True)

    st.text("Your Output image - Right Click on mouse to save as")
    sketch = sketchit(img, option)

    st.image(sketch, use_column_width=True)

