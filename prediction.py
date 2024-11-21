from model import GEN
import cv2
import numpy as np

def process(path):
	model = GEN()
	model.model.load_weights('tr.h5')
	image = cv2.imread(path)
	image = cv2.resize(image,(256,256))
	image = np.divide(image,255.0)
	image = np.expand_dims(image,axis=0)
	pred = model.model.predict(image)
	alpha = pred[0][0][0][0] + 0.12
	beta = pred[0][0][0][1]
	print(alpha,beta)
	image = cv2.imread(path)
	image = cv2.resize(image,(512,512))
	pred = cv2.convertScaleAbs(image,alpha = alpha,beta = beta*50)
	cv2.imshow('Result Image',pred)
	cv2.imshow('Original Image',image)
	cv2.waitKey(0)
	cv2.imwrite('Result Image.jpg',pred)


'''
#!/usr/bin/env python
__author__ = "Sreenivas Bhattiprolu"
__license__ = "Feel free to copy, I appreciate if you acknowledge Python for Microscopists"

# https://youtu.be/u_4d51bOsVs

"""
Unsharp mask enhances edges by subtracting an unsharp (smoothed) version of the image from the original.
Effectively making the filter a high pass filter. 
enhanced image = original + amount * (original - blurred)
Amount of sharpening can be controlled via scaling factor, a multiplication factor
for the sharpened signal. 
skimage uses Gaussian smoothing for image blurring therefore the radius parameter 
in the unsharp masking filter refers to the sigma parameter of the gaussian filter.
#This code shows that unsharp is nothing but original + amount *(original-blurred)
from skimage import io, img_as_float
from skimage.filters import unsharp_mask
from skimage.filters import gaussian
img = img_as_float(io.imread("images/einstein_blurred.jpg", as_gray=True))
gaussian_img = gaussian(img, sigma=1, mode='constant', cval=0.0)
img2 = (img - gaussian_img)*1.
img3 = img + img2
from matplotlib import pyplot as plt
plt.imshow(img3, cmap="gray")
"""

from skimage import io
from skimage.filters import unsharp_mask

img = io.imread("images/einstein_blurred.jpg")

#sharpened = unsharp_mask(image0, radius=1.0, amount=1.0)
unsharped_img = unsharp_mask(img, radius=3, amount=1)


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 12))
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap='gray')
ax1.title.set_text('Input Image')
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(unsharped_img, cmap='gray')
ax2.title.set_text('Unsharped Image')

plt.show()
'''