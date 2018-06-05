import pandas as pd
import numpy as np

from pylab import imshow, show, get_cmap
from numpy import random



#Z = random.random((5,5,3))   # Test data

# 4x3: rgb 
Z = [[[0.1,0.9,0.9],[0.9,0.1,0.9]],
[[0.1,0.9,0.8],[0.2,0.2,0.9]],
[[0.6,0.5,0.5],[0.4,0.4,0.4]],
[[0.1,0.1,0.1],[0.9,0.9,0.9]]]

imshow(Z, interpolation='nearest')
show()


def initialise(nx, ny):
	''' 
	nx: number of pixels in the x axis
	ny: number of pixels in the y axis
	'''
	

