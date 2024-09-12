import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import *

# region - Architectures -
# Residual Block -> ReLU
def ResBlock_ReLU(inputs):
    x = Conv2D(64, 3, padding="same", activation="ReLU")(inputs)
    x = Conv2D(64, 3, padding="same")(x)
    x = Add()([inputs, x])
    return x

# Residual Block -> LeakyReLU
def ResBlock_LeakyReLU(inputs):
    x = Conv2D(64, 3, padding="same", activation="LeakyReLU")(inputs)
    x = Conv2D(64, 3, padding="same")(x)
    x = Add()([inputs, x])
    return x

# Residual Block -> BatchNorm
def ResBlock_BatchNorm(inputs):
    x = Conv2D(64, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])
    return x

# Upsampling Block
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def UpsamplingBlock(inputs, factor=2, **kwargs):
    x = Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(inputs)
    x = Lambda(pixel_shuffle(scale=factor))(x)
    x = Conv2D(64 * (factor ** 2), 3, padding="same", **kwargs)(x)
    x = Lambda(pixel_shuffle(scale=factor))(x)
    return x
# endregion


# region - Models -
def EDSR_Vanilla(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = Input(shape=(None, None, 3))

    # Scaling Pixel Values
    x = Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock_ReLU(x_new)

    x_new = Conv2D(num_filters, 3, padding="same")(x_new)
    x = Add()([x, x_new])

    # Upsample
    x = UpsamplingBlock(x)

    x = Conv2D(3, 3, padding="same")(x)

    output_layer = Rescaling(scale=255)(x)
    return Model(input_layer, output_layer)

def EDSR_LeakyReLU(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = Input(shape=(None, None, 3))

    # Scaling Pixel Values
    x = Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock_LeakyReLU(x_new)

    x_new = Conv2D(num_filters, 3, padding="same")(x_new)
    x = Add()([x, x_new])

    # Upsample
    x = UpsamplingBlock(x)

    x = Conv2D(3, 3, padding="same")(x)

    output_layer = Rescaling(scale=255)(x)
    return Model(input_layer, output_layer)

def EDSR_ResBlockMod_ReLU(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = Input(shape=(None, None, 3))

    # Scaling Pixel Values
    x = Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock_ReLU(x_new)
        x_new = ReLU()(x_new)

    x_new = Conv2D(num_filters, 3, padding="same")(x_new)
    x = Add()([x, x_new])

    # Upsample
    x = UpsamplingBlock(x)

    x = Conv2D(3, 3, padding="same")(x)

    output_layer = Rescaling(scale=255)(x)
    return Model(input_layer, output_layer)

def EDSR_ResBlockMod_BatchNorm(num_filters, num_of_residual_blocks):
    # Flexible Inputs to input_layer
    input_layer = Input(shape=(None, None, 3))

    # Scaling Pixel Values
    x = Rescaling(scale=1.0 / 255)(input_layer)
    x = x_new = Conv2D(num_filters, 3, padding="same")(x)

    # 16 residual blocks
    for _ in range(num_of_residual_blocks):
        x_new = ResBlock_BatchNorm(x_new)

    x_new = Conv2D(num_filters, 3, padding="same")(x_new)
    x = Add()([x, x_new])

    # Upsample
    x = UpsamplingBlock(x)

    x = Conv2D(3, 3, padding="same")(x)

    output_layer = Rescaling(scale=255)(x)
    return Model(input_layer, output_layer)
# endregion

