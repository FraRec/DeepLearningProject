import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import *

# region - Architectures -
def normalize_01(x):
    return x / 255.0

def denormalize_m11(x):
    return (x + 1) * 127.5

# Upsample
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)
# endregion

# region - Blocks -
# Vanilla ResBlock
def residual_block(block_input, num_filters, last_layer=False, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)

    if(last_layer): x = Add(name="last_base_res_block")([block_input, x])
    else:           x = Add()([block_input, x])

    return x

# NoBatchNorm ResBlock
def residual_block_mod(block_input, num_filters, last_layer=False, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(block_input)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)

    if(last_layer): x = Add(name="last_base_res_block")([block_input, x])
    else:           x = Add()([block_input, x])

    return x

# DenseBlock
def residual_dense_block_mod(block_input, num_filters, last_layer=False, momentum=0.8):
  x1 = Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same")(block_input)
  y1 = tf.concat([block_input, x1], 3)

  x2 = Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same")(y1)
  y2 = tf.concat([block_input, x1, x2], 3)

  x3 = Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same")(y2)
  y3 = tf.concat([block_input, x1, x2, x3], 3)

  y = Conv2D(filters=num_filters, kernel_size=(1, 1), padding="same")(y3)
  return Add()([y, block_input])


# Build All ResBlocks
def BuildResBlocks(x, num_filters, num_res_blocks, last_base_layer_index, momentum=0.8):
  for i in range(num_res_blocks):
    x = residual_block(x, num_filters, last_base_layer_index == i)
  return x
# endregion

# region - Models -
# - Original Model -
def SRResNet_Original(scale=4, num_filters=64, num_res_blocks=16):
    # Input
    lr = Input(shape=(None, None, 3))

    # Normalizer
    x = Lambda(normalize_01)(lr)

    # First Conv
    x = Conv2D(num_filters, kernel_size=9, padding='same')(x)
    x = x_1 = PReLU(shared_axes=[1, 2], name="first_prelu")(x)

    # ResBlocks
    x = BuildResBlocks(x, num_filters, num_res_blocks, num_res_blocks//2)

    # Conv + Residual
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    # Upsample
    num_upsamples = 2
    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    # Last Conv
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    # De-Normalizer
    sr = Lambda(denormalize_m11)(x)

    return Model(lr, sr)

# - Models -
def SRResNet_Vanilla(model, num_res_blocks=8, scale=4, num_filters=64):
    x_1 = model.get_layer("first_prelu").output

    base_output = model.get_layer("last_base_res_block").output

    x = residual_block(base_output, num_filters)
    for i in range(num_res_blocks - 1):
        x = residual_block(x, num_filters)

    # Conv + Residual
    x = BatchNormalization()(x)
    x = Add(name="add_new")([x_1, x])

    # Upsample
    num_upsamples = 2
    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    # Last Conv
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    # De-Normalizer
    sr = Lambda(denormalize_m11)(x)

    return Model(model.input, sr)

def SRResNet_ResBlockMod(model, num_res_blocks=8, scale=4, num_filters=64):
    x_1 = model.get_layer("first_prelu").output

    base_output = model.get_layer("last_base_res_block").output

    x = residual_block_mod(base_output, num_filters)
    for i in range(num_res_blocks - 1):
        x = residual_block_mod(x, num_filters)

    # Conv + Residual
    x = BatchNormalization()(x)
    x = Add(name="add_new")([x_1, x])

    # Upsample
    num_upsamples = 2
    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    # Last Conv
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    # De-Normalizer
    sr = Lambda(denormalize_m11)(x)

    return Model(model.input, sr)

def SRResNet_DenseBlock(model, num_res_blocks=8, scale=4, num_filters=64):
    x_1 = model.get_layer("first_prelu").output

    base_output = model.get_layer("last_base_res_block").output

    x = residual_block_mod(base_output, num_filters)
    for i in range(num_res_blocks - 1):
        x = residual_dense_block_mod(x, num_filters)

    # Conv + Residual
    x = BatchNormalization()(x)
    x = Add(name="add_new")([x_1, x])

    # Upsample
    num_upsamples = 2
    for _ in range(num_upsamples):
        x = upsample(x, num_filters * 4)

    # Last Conv
    x = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(x)

    # De-Normalizer
    sr = Lambda(denormalize_m11)(x)

    return Model(model.input, sr)
# endregion

