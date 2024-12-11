import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import *

# region - Architectures -
# Upsampling Block
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def UpsamplingBlock(inputs, factor=2):
    x = Conv2D(64, 3, padding="same", activation="relu")(inputs)
    x = Lambda(pixel_shuffle(scale=factor))(x)
    return x
# endregion


# region - Blocks -
def ResDenseBlock(inputs):
  x1 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(inputs)
  y1 = tf.concat([inputs, x1], 3)

  x2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(y1)
  y2 = tf.concat([inputs, x1, x2], 3)

  x3 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same")(y2)
  y3 = tf.concat([inputs, x1, x2, x3], 3)

  y = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(y3)
  return Add()([y, inputs])

def ResDenseBlock_LeakyReLU(inputs):
  x1 = Conv2D(filters=64, kernel_size=(3, 3), activation="LeakyReLU", padding="same")(inputs)
  y1 = tf.concat([inputs, x1], 3)

  x2 = Conv2D(filters=64, kernel_size=(3, 3), activation="LeakyReLU", padding="same")(y1)
  y2 = tf.concat([inputs, x1, x2], 3)

  x3 = Conv2D(filters=64, kernel_size=(3, 3), activation="LeakyReLU", padding="same")(y2)
  y3 = tf.concat([inputs, x1, x2, x3], 3)

  y = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(y3)
  return Add()([y, inputs])

def ResDenseBlock_PReLU(inputs):
  x1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
  x1 = PReLU(shared_axes=[1, 2])(x1)
  y1 = tf.concat([inputs, x1], 3)

  x2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(y1)
  x2 = PReLU(shared_axes=[1, 2])(x2)
  y2 = tf.concat([inputs, x1, x2], 3)

  x3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(y2)
  x3 = PReLU(shared_axes=[1, 2])(x3)
  y3 = tf.concat([inputs, x1, x2, x3], 3)

  y = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(y3)
  return Add()([y, inputs])
# endregion


# region - Models -
def RDN_Vanilla():
  inputs = Input(shape=(None, None, 3))
  x = Rescaling(scale=1.0 / 255)(inputs)

  c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
  x  = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(c1)

  d1 = ResDenseBlock(x)
  d2 = ResDenseBlock(d1)
  d3 = ResDenseBlock(d2)

  d = tf.concat([d1, d2, d3], 3)
  x = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(d)
  x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)

  x = Add()([x, c1])

  x = UpsamplingBlock(x, factor=2)

  x = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(x)
  outputs = Rescaling(scale=255)(x)
  return Model(inputs, outputs)

def RDN_LeakyReLU():
  inputs = Input(shape=(None, None, 3))
  x = Rescaling(scale=1.0 / 255)(inputs)

  c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
  x  = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(c1)

  d1 = ResDenseBlock_LeakyReLU(x)
  d2 = ResDenseBlock_LeakyReLU(d1)
  d3 = ResDenseBlock_LeakyReLU(d2)

  d = tf.concat([d1, d2, d3], 3)
  x = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(d)
  x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)

  x = Add()([x, c1])

  x = UpsamplingBlock(x, factor=2)

  x = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(x)
  outputs = Rescaling(scale=255)(x)
  return Model(inputs, outputs)

def RDN_PReLU():
  inputs = Input(shape=(None, None, 3))
  x = Rescaling(scale=1.0 / 255)(inputs)

  c1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)
  x  = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(c1)

  d1 = ResDenseBlock_PReLU(x)
  d2 = ResDenseBlock_PReLU(d1)
  d3 = ResDenseBlock_PReLU(d2)

  d = tf.concat([d1, d2, d3], 3)
  x = Conv2D(filters=64, kernel_size=(1, 1), padding="same")(d)
  x = Conv2D(filters=64, kernel_size=(3, 3), padding="same")(x)

  x = Add()([x, c1])

  x = UpsamplingBlock(x, factor=2)

  x = Conv2D(filters=3, kernel_size=(3, 3), padding="same")(x)
  outputs = Rescaling(scale=255)(x)
  return Model(inputs, outputs)
# endregion

