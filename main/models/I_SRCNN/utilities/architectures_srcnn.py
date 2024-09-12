import tensorflow as tf

from keras import Model
from keras._tf_keras.keras.layers import *
from keras._tf_keras.keras.optimizers import *
from keras._tf_keras.keras.preprocessing.image import *

# region - Architectures -
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample_block(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return LeakyReLU()(x)
# endregion

# region - Models -
def SRCNN(input_dim):
  input = Input(shape=input_dim, name="Input")
  X = Conv2D(filters=128, kernel_size=9, activation="relu", padding="valid")(input)
  X = Conv2D(filters=64 , kernel_size=3, activation="relu", padding="same")(X)
  output = Conv2D(filters=3  , kernel_size=5, activation="linear", padding="valid")(X)
  return Model(input, output)

def SRCNN_LeakyReLU(input_dim):
  input = Input(shape=input_dim, name="Input")
  X = Conv2D(filters=128, kernel_size=9, activation="LeakyReLU", padding="valid")(input)
  X = Conv2D(filters=64 , kernel_size=3, activation="LeakyReLU", padding="same")(X)
  output = Conv2D(filters=3  , kernel_size=5, activation="linear", padding="valid")(X)
  return Model(input, output)

def SRCNN_Residual(input_dim, LABEL_SIZE):
  input = Input(shape=input_dim, name="Input")
  scaled = Resizing(LABEL_SIZE, LABEL_SIZE, interpolation="bilinear")(input)
  X = Conv2D(filters=128, kernel_size=9, activation="relu", padding="same")(scaled)
  X = Conv2D(filters=64 , kernel_size=3, activation="relu", padding="same")(X)
  residual = Conv2D(filters=3  , kernel_size=5, activation="linear", padding="same")(X)
  output = Add()([scaled, residual])
  return Model(input, output)

def SRCNN_Residual_LeakyReLU(input_dim, LABEL_SIZE):
  input = Input(shape=input_dim, name="Input")
  scaled = Resizing(LABEL_SIZE, LABEL_SIZE, interpolation="bilinear")(input)
  X = Conv2D(filters=128, kernel_size=9, activation="LeakyReLU", padding="same")(scaled)
  X = Conv2D(filters=64 , kernel_size=3, activation="LeakyReLU", padding="same")(X)
  residual = Conv2D(filters=3  , kernel_size=5, activation="linear", padding="same")(X)
  output = Add()([scaled, residual])
  return Model(input, output)

def SRCNN_PixelShuffle_LeakyReLU(input_dim):
  input = Input(shape=input_dim, name="Input")
  X = Conv2D(filters=64 , kernel_size=3, activation="LeakyReLU", padding="same")(input)
  X = Conv2D(filters=128, kernel_size=9, activation="LeakyReLU", padding="same")(X)
  X = upsample_block(X, 64 * 4)
  output = Conv2D(3, kernel_size=5, activation="linear", padding='same')(X)
  return Model(input, output)

def SRCNN_PixelShuffle_Residual_LeakyReLU(input_dim, LABEL_SIZE):
  input = Input(shape=input_dim, name="Input")
  scaled = Resizing(LABEL_SIZE, LABEL_SIZE, interpolation="bilinear")(input)
  X = Conv2D(filters=64 , kernel_size=5, activation="LeakyReLU", padding="same")(input)
  X = upsample_block(X, 32 * 4)
  X = Conv2D(3, kernel_size=5, activation="linear", padding='same')(X)
  output = Add()([scaled, X])
  return Model(input, output)
# endregion

'''
# - Architectures -
def calculate_content_loss(hr, pr, perceptual_model):
  # Resize Images
  hr = tf.image.resize(hr, size=(224, 224))
  pr = tf.image.resize(pr, size=(224, 224))

  # Convert to Array
  hr = tf.cast(hr, tf.float32)
  pr = tf.cast(pr, tf.float32)

  # Preprocess Images
  hr = preprocess_input(hr)
  pr = preprocess_input(pr)

  # Find Features
  pred_feature = perceptual_model(pr)
  hr_feature = perceptual_model(hr)

  vgg_loss = tf.math.reduce_mean(tf.math.squared_difference(hr_feature[0], pred_feature[0]))
  mse = tf.math.reduce_mean(tf.math.squared_difference(hr, pr))
  return vgg_loss + mse
'''

