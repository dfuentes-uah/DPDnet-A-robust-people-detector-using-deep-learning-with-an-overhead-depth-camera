from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Reshape
from tensorflow.keras.models import Model
import numpy as np
from scipy.ndimage import rotate
import tensorflow.keras
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as K
import scipy.io
import math
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D,Add,SeparableConv2D, MaxPooling2D,concatenate,ZeroPadding2D,Cropping2D,Dropout,Lambda,Reshape,Input,Concatenate, concatenate,Conv3D,BatchNormalization,Activation,UpSampling2D,Conv2DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model,Model
from skimage import data, img_as_float
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pylab as plt
import numpy as np
import random
import scipy
import cv2 as cv
from skimage.transform import rescale, resize, downscale_local_mean
from scipy import ndimage
from tensorflow.keras.models import Sequential, load_model,Model
from scipy import *
import os

batch_size = 15
epochs=40
relacion_aspecto=424/512
img_x=256
img_y=round(relacion_aspecto*img_x)
lengthdataset=40000
path='GOTPD_DATABASE/'

def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    im=im*(255/np.max(im))
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)
def encoding_identity_block(input_tensor, kernel_size, filters, stage, block):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = SeparableConv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x

def decoding_identity_block(input_tensor, kernel_size, filters, stage, block):

	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2DTranspose(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x

def encoding_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = SeparableConv2D(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = SeparableConv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def decoding_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1

	x=UpSampling2D(size=strides, data_format=None)(input_tensor)
	x = SeparableConv2D(filters1, (1, 1))(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters2, kernel_size,padding='same')(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = Activation('relu')(x)

	x = SeparableConv2D(filters3, (1,1))(x)
	x = BatchNormalization(axis=bn_axis)(x)

	shortcut = UpSampling2D(size=strides, data_format=None)(input_tensor)
	shortcut = SeparableConv2D(filters3, (1, 1) )(shortcut)
	shortcut = BatchNormalization(axis=bn_axis)(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def refunit(divider,ch):

    image_input = Input(shape=(int(img_y/divider), int(img_x/divider), ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
    x=ZeroPadding2D(padding=(0,1),data_format=None)(x)

    x = UpSampling2D(size=(3, 3))(x)
    x = Cropping2D(cropping=((2, 2), (1, 1)), data_format=None)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
    x = Activation('sigmoid')(x)
    modelo = Model(inputs=image_input, outputs=x)
    modelo.summary()
    return modelo

def load_valdata(divider,canales):
    valinput=[]
    valoutput=[]
    multiplier=6
    counter=0
    l1=["validacion/imagenes/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png","validacion/imagenes/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png","validacion/imagenes/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png","validacion/imagenes/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png"]
    l2=["validacion/gaussianas/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png","validacion/gaussianas/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png","validacion/gaussianas/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png","validacion/gaussianas/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png"]
    l3=[741,509,920,868]
    while 1:
        valinput = []
        valoutput=  []
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
             ind=np.uint16(rand()*4)
             j=np.uint16(rand()*(l3[ind]-5))+1
             img_path = path+l1[ind] % (j)
             imgc = scipy.misc.imread(img_path, mode='F')
             imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
             xc = image.img_to_array(imgc)
             xc = xc / 65536
             if(canales is 3):
                xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
             valinput.append(xc)

             img_path = path+l2[ind] % (j)
             imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
             xc = image.img_to_array(imgc)
             xc = cv.blur(xc, (3, 3))
             xc = np.expand_dims(xc, axis=2)
             xc = xc / 255
             valoutput.append(xc)
        valinput=np.array(valinput)
        valoutput=np.array(valoutput)
        yield valinput,[valoutput,valoutput]

def TrainGen(divider,canales):
    counter=0
    while 1:
        X = []
        Y=  []
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
            j=math.floor(rand()*(lengthdataset-5))+1
            img_path = path+"imagenes/image%05d.png" % (j)
            imgc = scipy.misc.imread(img_path, mode='F')
            imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
            xc = image.img_to_array(imgc)
            xc = xc / 65536
            if (canales is 3):
                xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
            X.append(xc)

            img_path = path+"gaussianas/image%05d.png" % (j)
            imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
            xc = image.img_to_array(imgc)
            xc = cv.blur(xc, (3, 3))
            xc = np.expand_dims(xc, axis=2)
            xcaux = np.copy(xc)
            xcaux = abs(xcaux - 255)
            xc = xc / 255
            Y.append(xc)
        X = np.array(X)
        Y= np.array(Y)
        counter = counter + 1
        yield X,[Y,Y]

divider=1
canales=1
image_input = Input(shape=(int(img_y / divider), int(img_x / divider), 1))
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
x = BatchNormalization(axis=3, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3))(x)

x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))

x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')

x = encoding_conv_block(x, 3, [256, 256, 1024], stage=4, block='a')

x = decoding_conv_block(x, 3, [1024, 1024, 256], stage=5, block='a', strides=(1, 1))

x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')

x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
x = Cropping2D(cropping=((0, 0), (0, 1)), data_format=None)(x)

x = UpSampling2D(size=(3, 3))(x)
x = Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='same', name='co')(x)
x = Cropping2D(cropping=((2, 2), (1, 1)), data_format=None)(x)
x = BatchNormalization(axis=3, name='bn_c1')(x)
x = Activation('relu')(x)
x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
x = Activation('sigmoid')(x)
x2=tensorflow.keras.backend.concatenate([x,image_input],axis=-1)
refinement1 = refunit(divider, canales + 1)
x2=refinement1(x2)
model = Model(inputs=image_input, outputs=[x,x2])
model.summary()
check = tensorflow.keras.callbacks.ModelCheckpoint('DPDnet.h5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='auto', period=1)
model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.0),loss=['mse','mse'])
#[valinput,valoutput]=load_valdata(divider,canales)
trainlossactual = model.fit_generator(TrainGen(divider,canales), callbacks=[check],steps_per_epoch=math.floor(lengthdataset / batch_size),validation_data=load_valdata(divider,canales),validation_steps=1000, epochs=epochs, verbose=1)








