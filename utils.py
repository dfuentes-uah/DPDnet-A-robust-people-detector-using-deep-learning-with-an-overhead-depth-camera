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

batch_size = 7
pseudobatch_size=2000
epochs = 100
depthlist=[]
rgblist=[]
relacion_aspecto=424/512
img_x=256
img_y=round(relacion_aspecto*img_x)
trainpercentaje=0.99
lengthdataset=round(40760*trainpercentaje)#round(39976*trainpercentaje)

def load_batch(batch_size,counter):
    batcharray=[]
    batchoutput=[]
    for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
        #img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/home/geintra/TOF_COUNTING/imagenes_mix/image%05d.png" % (j)
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc, (img_y, img_x, 1), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        #xc = ndimage.median_filter(xc, 7)
        batcharray.append(xc)

        # img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/home/geintra/TOF_COUNTING/mascaras/mask%05d.png" % (j)
        imgc = image.load_img(img_path, grayscale=True, target_size=(img_y, img_x, 1))
        xc = image.img_to_array(imgc)
        xc = cv.blur(xc, (3, 3))
        xc = np.expand_dims(xc, axis=2)
        xcaux = np.copy(xc)
        xcaux = abs(xcaux - 255)
        #xc = np.concatenate([xc, xcaux], axis=2)
        xc = xc / 255
        batchoutput.append(xc)

    batcharray=np.array(batcharray)
    batchoutput=np.array(batchoutput)
    comprobar = 0
    if (comprobar is 1):
        for j in range(1, 918, 50):
            plt.figure(1)
            plt.imshow(batcharray[j, :, :, :])
            plt.figure(2)
            image1 = to_rgb3(batchoutput[j, :, :, 0])  # cv.cvtColor(valoutput[j, :, :, :], cv.COLOR_GRAY2RGB)
            plt.imshow(image1)
            plt.figure(3)
            image1 = to_rgb3(batchoutput[j, :, :, 1])  # cv.cvtColor(valoutput[j, :, :, :], cv.COLOR_GRAY2RGB)
            plt.imshow(image1)
            plt.show()
    return batcharray,batchoutput

def load_TVHEADTrain(divider,canales):
    input=[]
    output=[]
    #gt=[]
    path="/media/david/Datos/TOF/BDTOF/tvheads/gaussians/input/"
    import os
    listing = os.listdir(path)
    for file in listing:
         dif=0
         off=0#35-dif
         img_path = "/media/david/Datos/TOF/BDTOF/tvheads/gaussians/input/"+file
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc /65535.0#/ np.max(xc)
         #xc[xc == 0.0] =3500
         xc[:off,:]=0
         xc[len(xc[:,0])-off:len(xc[:,0]), :] = 0
         xc[:, :off] = 0
         xc[:,len(xc[0, :]) - off:len(xc[0, :])] = 0
         input.append(xc)

         off2 = 0#40-dif
         img_path = "/media/david/Datos/TOF/BDTOF/tvheads/gaussians/output/"+file
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc / 255.0
         xc[:off2,:]=0
         xc[len(xc[:,0])-off2:len(xc[:,0]), :] = 0
         xc[:, :off2] = 0
         xc[:,len(xc[0, :]) - off2:len(xc[0, :])] = 0
         output.append(xc)

    input=np.array(input)
    output=np.array(output)
    # plt.figure(1)
    # plt.imshow(input[0, :, :, :])
    # plt.show()
    #gt=np.array(gt)
    comprobar=0
    return input,output
def load_TVHEAD(divider,canales):
    input=[]
    output=[]
    #gt=[]
    path="/media/david/Datos/TOF/BDTOF/tvheads/16"
    import os
    listing = os.listdir(path)
    for file in listing:
         dif=0
         off=0-dif
         img_path = "/media/david/Datos/TOF/BDTOF/tvheads/16/"+file
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc /3500.0#/ np.max(xc)
         #xc[xc == 0.0] =3500
         xc[:off,:]=0
         xc[len(xc[:,0])-off:len(xc[:,0]), :] = 0
         xc[:, :off] = 0
         xc[:,len(xc[0, :]) - off:len(xc[0, :])] = 0


         #xc = ndimage.median_filter(xc, 7)
         img_path = "/media/david/Datos/TOF/BDTOF/tvheads/image1.png"
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         background = image.img_to_array(imgc)/65535.0
         # xc[xc>0.65]=background[xc>0.65]
         # xc[xc == 0.0] = background[xc == 0.0]
         #xc = ndimage.median_filter(xc, 3)
         input.append(xc)

         off2 = 0-dif
         img_path = "/media/david/Datos/TOF/BDTOF/tvheads/mask/"+file[:len(file)-4]+"_mask.png"
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc / np.max(xc)
         # xc[:off2,:]=0
         # xc[len(xc[:,0])-off2:len(xc[:,0]), :] = 0
         # xc[:, :off2] = 0
         # xc[:,len(xc[0, :]) - off2:len(xc[0, :])] = 0
         output.append(xc)

    input=np.array(input)
    output=np.array(output)
    # plt.figure(1)
    # plt.imshow(input[0, :, :, :])
    # plt.show()
    #gt=np.array(gt)
    comprobar=0
    return input,output
def load_ZHANGS_DATABASE(divider,canales):
    input=[]
    #gt=[]
    for j in range(1, 2380):
         img_path = "/media/david/Datos/BDTOF/paperzhang2012/HeadData-CBSR/dataset1/DEPTH_ADAPTADO_CNN/image%d.png" % (j)
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc / 65536
         if(canales is 3):
            xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
         # xc = ndimage.median_filter(xc, 7)
         input.append(xc)
    input=np.array(input)
    # plt.figure(1)
    # plt.imshow(input[0, :, :, :])
    # plt.show()
    #gt=np.array(gt)
    comprobar=0
    return input

def load_MVIA(divider,canales):
    input=[]
    #gt=[]
    for j in range(1, 4380):
        img_path = "/media/david/Datos/BDTOF/MIVIA-PeopleCounting/videos/depth/DEPTH_CNN_DIG2/image2616.png"
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc,(int(img_y/divider), int(img_x/divider)), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        if (canales is 3):
            xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        #xc = ndimage.median_filter(xc, 7)
        input.append(xc)
    input=np.array(input)
    # plt.figure(1)
    # plt.imshow(input[0, :, :, :])
    # plt.show()
    #gt=np.array(gt)
    comprobar=0
    return input

def to_rgb16(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    im=im*(65536)
    return np.asarray(np.dstack((im, im, im)), dtype=np.float32)
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
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2DTranspose(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2DTranspose(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2DTranspose(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x

def decoding_separable_conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,2)):
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	#x = Conv2DTranspose(filters1, (1, 1), strides=strides,name=conv_name_base + '2a')(input_tensor)
	x=UpSampling2D(size=strides, data_format=None)(input_tensor)
	x = SeparableConv2D(filters1, (1, 1))(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = Activation('relu')(x)

	#x = Conv2DTranspose(filters2, kernel_size, padding='same',name=conv_name_base + '2b')(x)
	#x = UpSampling2D(size=(2, 2), data_format=None)(x)
	x = SeparableConv2D(filters2, kernel_size,padding='same')(x)
	x = BatchNormalization(axis=bn_axis)(x)
	x = Activation('relu')(x)

	#x = Conv2DTranspose(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = SeparableConv2D(filters3, (1,1))(x)
	x = BatchNormalization(axis=bn_axis)(x)

	#shortcut = Conv2DTranspose(filters3, (1, 1), strides=strides,name=conv_name_base + '1')(input_tensor)
	shortcut = UpSampling2D(size=strides, data_format=None)(input_tensor)
	shortcut = SeparableConv2D(filters3, (1, 1) )(shortcut)
	shortcut = BatchNormalization(axis=bn_axis)(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x
def refunit(ch):
    # image_input = tf.placeholder(tf.float32, shape=(height, width, 3))
    image_input = Input(shape=(img_y, img_x, ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    #x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')
    x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='b')
    x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='c')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='d')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
    x=Cropping2D(cropping=((1,0),(0,0)),data_format=None)(x)
    x=ZeroPadding2D(padding=(0,1),data_format=None)(x)
    x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='b')
    x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='c')

    x = UpSampling2D(size=(3, 3))(x)
    x = Cropping2D(cropping=((0, 1), (1, 1)), data_format=None)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
    x = Activation('sigmoid')(x)
    modelo = Model(inputs=image_input, outputs=x)
    modelo.summary()
    return modelo

def microrefunit(divider,ch):
    # image_input = tf.placeholder(tf.float32, shape=(height, width, 3))
    image_input = Input(shape=(int(img_y/divider), int(img_x/divider), ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    #x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    #x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='b')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='c')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='d')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
    x=Cropping2D(cropping=((0,0),(0,0)),data_format=None)(x)
    x=ZeroPadding2D(padding=(0,0),data_format=None)(x)
    #x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='b')
    #x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='c')

    x = UpSampling2D(size=(3, 3))(x)
    x = Cropping2D(cropping=((1, 1), (2, 2)), data_format=None)(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', name='c8o')(x)
    x = Activation('sigmoid')(x)
    modelo = Model(inputs=image_input, outputs=x)
    modelo.summary()
    return modelo
def microrefunitv2(divider,ch):
    # image_input = tf.placeholder(tf.float32, shape=(height, width, 3))
    image_input = Input(shape=(int(img_y/divider), int(img_x/divider), ch))
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(image_input)
    x = BatchNormalization(axis=3, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3))(x)

    x = encoding_conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    #x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    #x = encoding_identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = encoding_conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    #x = encoding_identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = decoding_conv_block(x, 3, [512, 512, 128], stage=6, block='a')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='b')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='c')
    #x = decoding_identity_block(x, 3, [512, 512, 128], stage=6, block='d')

    x = decoding_conv_block(x, 3, [256, 256, 64], stage=7, block='a')
    #x=Cropping2D(cropping=((0,0),(0,0)),data_format=None)(x)
    x=ZeroPadding2D(padding=(0,1),data_format=None)(x)
    #x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='b')
    #x = decoding_identity_block(x, 3, [256, 256, 64], stage=7, block='c')

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
    for j in range(1, 741,multiplier):
         img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/imagenes/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png" % (j)
         imgc = scipy.misc.imread(img_path, mode='F')
         imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
         xc = image.img_to_array(imgc)
         xc = xc / 65536
         if(canales is 3):
            xc=np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
         #xc = ndimage.median_filter(xc, 7)
         valinput.append(xc)

         # img_path = "shape_template/images/image%d.png" % (j)
         img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/gaussianas/seq-P01-M04-A0002-G00-C00-S0101/image%04d.png" % (j)
         imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
         xc = image.img_to_array(imgc)
         xc = cv.blur(xc, (3, 3))
         xc = np.expand_dims(xc, axis=2)
         xcaux = np.copy(xc)
         xcaux = abs(xcaux-255)
         #xc=np.concatenate([xc, xcaux],axis=2)
         xc = xc / 255
         valoutput.append(xc)
    for j in range(1, 509,multiplier):
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/imagenes/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png" % (j)
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        if (canales is 3):
            xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        #xc = ndimage.median_filter(xc, 7)
        valinput.append(xc)
        # img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/gaussianas/seq-P05-M04-A0001-G03-C00-S0030/image%04d.png" % (j)
        imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
        xc = image.img_to_array(imgc)
        xc = cv.blur(xc, (3, 3))
        xc = np.expand_dims(xc, axis=2)
        xcaux = xc
        xcaux = abs(xcaux - 255)
        #xc = np.concatenate([xc, xcaux], axis=2)
        xc = xc / 255
        valoutput.append(xc)
    for j in range(1, 920,multiplier):
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/imagenes/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png" % (j)
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        if (canales is 3):
            xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        #xc = ndimage.median_filter(xc, 7)
        valinput.append(xc)
        # img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/gaussianas/seq-P00-M02-A0032-G00-C00-S0037/image%04d.png" % (j)
        imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
        xc = image.img_to_array(imgc)
        xc = cv.blur(xc, (3, 3))
        xc = np.expand_dims(xc, axis=2)
        xcaux = xc
        xcaux = abs(xcaux - 255)
        #xc = np.concatenate([xc, xcaux], axis=2)
        xc = xc / 255
        valoutput.append(xc)
    for j in range(1, 868,multiplier):
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/imagenes/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png" % (j)
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        if (canales is 3):
            xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        #xc = ndimage.median_filter(xc, 7)
        valinput.append(xc)
        # img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/validacion/gaussianas/seq-P00-M02-A0032-G00-C00-S0036/image%04d.png" % (j)
        imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
        xc = image.img_to_array(imgc)
        xc = cv.blur(xc, (3, 3))
        xc = np.expand_dims(xc, axis=2)
        xcaux = xc
        xcaux = abs(xcaux - 255)
        #xc = np.concatenate([xc, xcaux], axis=2)
        xc = xc / 255
        valoutput.append(xc)
    #seq-P00-M02-A0032-G00-C00-S0037
    valinput=np.array(valinput)
    valoutput=np.array(valoutput)
    comprobar=0
    if(comprobar is 1):
        for j in range(1, 918,50):
           plt.figure(1)
           plt.imshow(valinput[j,:,:,:])
           plt.figure(2)
           image1 = to_rgb3(valoutput[j, :, :, 0])#cv.cvtColor(valoutput[j, :, :, :], cv.COLOR_GRAY2RGB)
           plt.imshow(image1)
           plt.figure(3)
           image1 = to_rgb3(valoutput[j, :, :, 1])  # cv.cvtColor(valoutput[j, :, :, :], cv.COLOR_GRAY2RGB)
           plt.imshow(image1)
           plt.show()
    #valoutput=output[lengthdataset-1:round((lengthdataset/trainpercentaje))-1,:,:]
    return valinput,valoutput

from scipy import *
def TrainGen(divider,canales):
    counter=0
    while 1:
        X = []
        Y=  []
        Y2=[]
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
            #img_path = "shape_template/images/image%d.png" % (j)
            j=math.floor(rand()*(lengthdataset-5))+1
            img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/imagenes/image%05d.png" % (j)
            imgc = scipy.misc.imread(img_path, mode='F')
            imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
            xc = image.img_to_array(imgc)
            xc = xc / 65536
            if (canales is 3):
                xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
            X.append(xc)
            # img_path = "shape_template/images/image%d.png" % (j)
            # img_path = "shape_template/images/image%d.png" % (j)
            img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/gaussianas/image%05d.png" % (j)
            imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
            xc = image.img_to_array(imgc)
            xc = cv.blur(xc, (3, 3))
            xc = np.expand_dims(xc, axis=2)
            xcaux = np.copy(xc)
            xcaux = abs(xcaux - 255)
            # xc = np.concatenate([xc, xcaux], axis=2)
            xc = xc / 255
            Y.append(xc)
        X = np.array(X)
        Y= np.array(Y)
        # Y = ptos3d[batch_size * (counter):batch_size * (counter + 1)]
        counter = counter + 1#int((rand()*lengthdataset)/batch_size)
        yield X,[Y,Y]
        #return X,[Y,Y]

def AUTOENCODERGen(divider,canales):
    counter=0
    while 1:
        X = []
        Y=  []
        Y2=[]
        for j in range(batch_size*counter+1, batch_size*(counter+1)+1):
            #img_path = "shape_template/images/image%d.png" % (j)
            j=math.floor(rand()*(lengthdataset-5))+1
            img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/imagenes/image%05d.png" % (j)
            imgc = scipy.misc.imread(img_path, mode='F')
            imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
            xc = image.img_to_array(imgc)
            xc = xc / 65536
            fliplr1=0
            flipud1=0
            if (canales is 3):
                xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
            if(rand()>0.25):
                xc=np.fliplr(xc)
                fliplr1=1
            if (rand() > 0.25):
                xc = np.flipud(xc)
                flipud1 = 1
            Y.append(xc)
            # img_path = "shape_template/images/image%d.png" % (j)
            # img_path = "shape_template/images/image%d.png" % (j)
            img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/gaussianas/image%05d.png" % (j)
            imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
            xc = image.img_to_array(imgc)
            xc = cv.blur(xc, (3, 3))
            xc = np.expand_dims(xc, axis=2)
            xcaux = np.copy(xc)
            xcaux = abs(xcaux - 255)
            # xc = np.concatenate([xc, xcaux], axis=2)
            xc = xc / 255
            if(fliplr1==1):
                xc=np.fliplr(xc)
                fliplr1=0
            if (flipud1==1):
                xc = np.flipud(xc)
                flipud1=0
            X.append(xc)
        X = np.array(X)
        Y= np.array(Y)
        # Y = ptos3d[batch_size * (counter):batch_size * (counter + 1)]
        counter = counter + 1#int((rand()*lengthdataset)/batch_size)
        yield X,Y
        #return X,Y
def AUTOENCODERGenv2(divider,canales):
    counter=0
    X = []
    Y=  []
    Y2=[]
    batch_size=7
    for j in range(0, 200):
        #img_path = "shape_template/images/image%d.png" % (j)
        j=math.floor(rand()*(lengthdataset-5))+1
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/imagenes/image%05d.png" % (j)
        imgc = scipy.misc.imread(img_path, mode='F')
        imgc = scipy.misc.imresize(imgc, (int(img_y/divider), int(img_x/divider), 1), interp='bilinear', mode='F')
        xc = image.img_to_array(imgc)
        xc = xc / 65536
        fliplr1=0
        flipud1=0
        if (canales is 3):
            xc = np.asarray(np.dstack((xc, xc, xc)), dtype=np.float64)
        if(rand()>0.25):
            xc=np.fliplr(xc)
            fliplr1=1
        if (rand() > 0.25):
            xc = np.flipud(xc)
            flipud1 = 1
        Y.append(xc)
        # img_path = "shape_template/images/image%d.png" % (j)
        # img_path = "shape_template/images/image%d.png" % (j)
        img_path = "/media/david/Datos/TOF/MIX_GAUSSIANAS_NUEVO/gaussianas/image%05d.png" % (j)
        imgc = image.load_img(img_path, grayscale=True, target_size=(int(img_y/divider), int(img_x/divider), 1))
        xc = image.img_to_array(imgc)
        xc = cv.blur(xc, (3, 3))
        xc = np.expand_dims(xc, axis=2)
        xcaux = np.copy(xc)
        xcaux = abs(xcaux - 255)
        # xc = np.concatenate([xc, xcaux], axis=2)
        xc = xc / 255
        if(fliplr1==1):
            xc=np.fliplr(xc)
            fliplr1=0
        if (flipud1==1):
            xc = np.flipud(xc)
            flipud1=0
        X.append(xc)
    X = np.array(X)
    Y= np.array(Y)
    # Y = ptos3d[batch_size * (counter):batch_size * (counter + 1)]
    counter = counter + 1#int((rand()*lengthdataset)/batch_size)
    return X,Y