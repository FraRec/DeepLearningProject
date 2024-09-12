import tensorflow as tf

from tf_keras import Model
from tf_keras.layers import *
from tf_keras.applications.vgg19 import preprocess_input
from tf_keras.losses import mean_squared_error, binary_crossentropy


# region - Architectures -
# Normalize
def normalize_01(x):
    return x / 255.0

def denormalize_m11(x):
    return (x + 1) * 127.5

def normalize_m11(x):
    """Normalizes RGB images to [-1, 1]."""
    return x / 127.5 - 1


# Upsample
def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample(x_in, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = Lambda(pixel_shuffle(scale=2))(x)
    return PReLU(shared_axes=[1, 2])(x)


# Training Loop Functions
def calculate_content_loss(hr, sr, perceptual_model):
    sr = preprocess_input(sr)
    hr = preprocess_input(hr)
    sr_features = perceptual_model(sr) / 12.75
    hr_features = perceptual_model(hr) / 12.75
    return mean_squared_error(hr_features, sr_features)

def calculate_generator_loss(sr_out):
    return binary_crossentropy(tf.ones_like(sr_out), sr_out)

def calculate_discriminator_loss(hr_out, sr_out):
    hr_loss = binary_crossentropy(tf.ones_like(hr_out), hr_out)
    sr_loss = binary_crossentropy(tf.zeros_like(sr_out), sr_out)
    return hr_loss + sr_loss
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


# Discriminator Block
def discriminator_block(x_in, num_filters, strides=1, batchnorm=True, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    if batchnorm:
        x = BatchNormalization(momentum=momentum)(x)
    return LeakyReLU(alpha=0.2)(x)

# Build All ResBlocks
def BuildResBlocks(x, num_filters, num_res_blocks, last_base_layer_index, momentum=0.8):
  for i in range(num_res_blocks):
    x = residual_block(x, num_filters, last_base_layer_index == i)
  return x
# endregion

# region - Training Model -

class TrainingClassGAN(Model):
  def __init__(self, generator, discriminator, *args, **kwargs):
    # Pass Through args and kwargs to base class
    super().__init__(*args, **kwargs)

    # Create attribute for Gen and Dis
    self.generator = generator
    self.discriminator = discriminator

  def compile(self, g_opt, d_opt, g_loss, d_loss, *args, **kwargs):
    # Compile with Base Class
    super().compile(*args, **kwargs)

    # Create attributes for losses and optimizers
    self.g_opt = g_opt
    self.d_opt = d_opt
    self.g_loss = g_loss
    self.d_loss = d_loss

  def train_step(self, lr, hr):
    # Train the Discriminator & Generator
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        hr = tf.cast(hr, tf.float32)            # -> High Resolution Image
        lr = tf.cast(lr, tf.float32)            # --> Low Resolution Image
        sr = self.generator(lr, training=True)  # -------> Predicted Image

        # Pass the Real and Fake Images to the Discriminator Model
        hr_output = self.discriminator(hr, training=True)
        sr_output = self.discriminator(sr, training=True)

        # Calculate Loss
        con_loss = calculate_content_loss(hr, sr)
        gen_loss = calculate_generator_loss(sr_output)
        perc_loss = con_loss + 0.001 * gen_loss
        disc_loss = calculate_discriminator_loss(hr_output, sr_output)

    # Apply Backpropagation - nn Learn
    gradients_of_generator = gen_tape.gradient(perc_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

    self.g_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
    self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    return perc_loss, disc_loss

# endregion

# region - Original Models -
# - Generator Model -
def Generator(scale=4, num_filters=64, num_res_blocks=16):
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

# - Discriminator Model -
def Discriminator(hr_crop_size=96):
    x_in = Input(shape=(hr_crop_size, hr_crop_size, 3))
    x = Lambda(normalize_m11)(x_in)

    x = discriminator_block(x, 64, batchnorm=False)
    x = discriminator_block(x, 64, strides=2)

    x = discriminator_block(x, 128)
    x = discriminator_block(x, 128, strides=2)

    x = discriminator_block(x, 256)
    x = discriminator_block(x, 256, strides=2)

    x = discriminator_block(x, 512)
    x = discriminator_block(x, 512, strides=2)

    x = Flatten()(x)

    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(x_in, x)
# endregion

# region - Custom Models -
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

