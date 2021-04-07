from __future__ import print_function
import numpy as np
from scipy.ndimage import rotate
import scipy.io
import math
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
import random
import scipy
import cv2
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage
import os
from numpy import *
import imageio
refPt = []
sequence_length=100

def gaussian(px, py, desv=30./2.5):
    x=np.linspace(1.0, 256.0, num=256)
    y = np.linspace(1.0, 212.0, num=212)
    X, Y = np.meshgrid(x, y)
    px = np.float(px);
    py = np.float(py);
    z = (exp(-(np.square((X - px)/desv)/ 2) - (np.square((Y - py)/desv)/ 2)))
    z = z * 255
    z=np.expand_dims(z,axis=2)
    z = np.uint8(z)

    return z

def click_and_crop(event, x, y, flags, param):
    global refPt, gaussina,arrayn

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))

        arrayn=np.array(refPt)
        if 'gaussina' in locals() or globals():
            print('exist')
        else:
            gaussina=np.zeros((212,256,1))
        for f in range(0,np.int(len(arrayn)/2),1):
            gaussina=np.amax(np.concatenate((gaussian(arrayn[f,0], arrayn[f,1]),gaussina),axis=2), axis=2)
            gaussina=np.expand_dims(gaussina,axis=2)

        cv2.imshow("Gaussians_map", np.uint8(gaussina))
    cv2.imshow("Input_image", image)

cv2.namedWindow("Input_image")
cv2.setMouseCallback("Input_image", click_and_crop)

for j in range(1,sequence_length,1):
    folder="/media/david/Datos/DPD_NET/MIX_GAUSSIANAS/imagenes/" #Change the path here for your database
    img_path = folder+"image%05d.png" % (j)
    image = imageio.imread(img_path)
    image = cv2.resize(image, (256, 212))
    image=np.uint8(image*(255./np.max(image)))
    image=np.expand_dims(image,axis=2)
    gaussina=image.copy()*0
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
    import os
    folder=folder+'gaussianas/'
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)
    #cv2.imwrite(folder+"depth%06d.png" % (j),gaussina)



