# laser_sketch

 I made my own LightRoom type filter app with streamlit and python , 
 and deployed to Heroku it can be found here 
 https://laser-sketch.herokuapp.com/
 
 Platform used Python Anaconda 3.8 for Windows, the Streamlit app.
 
 Install instructions for streamlit is here
 https://docs.streamlit.io/en/0.78.0/installation.html
 
 Following dependinces below will need to be loaded ither through pip
 or conda install.
 
 import numba  # We added these two lines for a 500x speedup
 import cv2
 import streamlit as st
 import numpy as np
 from PIL import Image, ImageEnhance
 import math
 import sys, PIL.Image
 import numba.cuda
 import scipy
 
When finished you can test locally 
open an Anconda prompt cd into
.spyder-py3 test your streamlit install.
streamlit hello
This will start the stremlit demo.

To test laser_skectch.py enter
the following prompt

streamlit run laser_sketch.py

Enjoy


 
