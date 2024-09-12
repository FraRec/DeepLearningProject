import tensorflow as tf

from tf_keras.layers import *
from tf_keras.models import Model
from tf_keras.applications.vgg19 import preprocess_input
from tf_keras.applications.vgg19 import VGG19
from tf_keras.preprocessing.image import *

# region - Architectures -
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample_block(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)
# endregion

# region - Models -
def VDSR_Vanilla(input_dim, l, LABEL_SIZE):
    #Define input layer
    LR = Input(shape=input_dim, name='input')

    #Rescale Input
    resized = Resizing(LABEL_SIZE, LABEL_SIZE, "bilinear")(LR)

    #First convolution
    X = ZeroPadding2D()(resized)
    X = Conv2D(64,(3,3), name='CONV1')(X)
    X = ReLU()(X)

    #Repeat convolution layers untill last layer
    for i in range(l-2):
        X = ZeroPadding2D()(X)
        X = Conv2D(64, (3,3), name='CONV%i' % (i+2))(X)
        X = ReLU()(X)

    #Final layer, output is residual image
    X = ZeroPadding2D()(X)
    residual = Conv2D(1, (3,3), name='CONV%i' % l)(X)

    #Add residual to LR
    out = Add()([resized, residual])

    return Model(LR, out)

def VDSR_LeakyReLU(input_dim, l, LABEL_SIZE):
    #Define input layer
    LR = Input(shape=input_dim, name='input')

    #Rescale Input
    resized = Resizing(LABEL_SIZE, LABEL_SIZE, "bilinear")(LR)

    #First convolution
    X = ZeroPadding2D()(resized)
    X = Conv2D(64,(3,3), name='CONV1')(X)
    X = LeakyReLU()(X)

    #Repeat convolution layers untill last layer
    for i in range(l-2):
        X = ZeroPadding2D()(X)
        X = Conv2D(64, (3,3), name='CONV%i' % (i+2))(X)
        X = LeakyReLU()(X)

    #Final layer, output is residual image
    X = ZeroPadding2D()(X)
    residual = Conv2D(1, (3,3), name='CONV%i' % l)(X)

    #Add residual to LR
    out = Add()([resized, residual])

    return Model(LR, out)

def VDSR_Vanilla_PixelShuffle(input_dim, l, LABEL_SIZE):
    #Define input layer
    LR = Input(shape=input_dim, name='input')

    #Rescale Input
    resized = Resizing(LABEL_SIZE, LABEL_SIZE, "bilinear")(LR)

    #First convolution
    X = ZeroPadding2D()(LR)
    X = Conv2D(64,(3,3), name='CONV1')(X)
    X = ReLU()(X)

    #Repeat convolution layers untill last layer
    for i in range(round((l-2) / 2)):
        X = ZeroPadding2D()(X)
        X = Conv2D(64, (3,3), name='CONV%i' % (i+2))(X)
        X = ReLU()(X)

    for i in range(4, round((l-2) / 2) + 4):
        X = ZeroPadding2D()(X)
        X = Conv2D(64, (3,3), name='CONV%i' % (i+2))(X)
        X = ReLU()(X)

    X = upsample_block(X, 32 * 4)

    #Final layer, output is residual image
    X = ZeroPadding2D()(X)
    residual = Conv2D(3, (3,3), name='CONV%i' % l)(X)

    #Add residual to LR
    out = Add()([resized, residual])

    return Model(LR, out)
# endregion




