import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tf_keras.preprocessing.image import img_to_array

# region - Data Augmentation -
def flip_left_right(lowres_img, highres_img):
    """Flips Images to left and right."""

    # Outputs random values from a uniform distribution in between 0 to 1
    rn = tf.random.uniform(shape=(), maxval=1)
    # If rn is less than 0.5 it returns original lowres_img and highres_img
    # If rn is greater than 0.5 it returns flipped image
    return tf.cond(
        rn < 0.5,
        lambda: (lowres_img, highres_img),
        lambda: (
            tf.image.flip_left_right(lowres_img),
            tf.image.flip_left_right(highres_img),
        ),
    )

def random_rotate(lowres_img, highres_img):
    """Rotates Images by 90 degrees."""

    # Outputs random values from uniform distribution in between 0 to 4
    rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    # Here rn signifies number of times the image(s) are rotated by 90 degrees
    return tf.image.rot90(lowres_img, rn), tf.image.rot90(highres_img, rn)

def random_crop(lowres_img, highres_img, hr_crop_size=96, scale=2):
    """Crop images.
    low resolution images: 48x48
    high resolution images: 96x96
    """
    lowres_crop_size = hr_crop_size // scale  # 96//2=48
    lowres_img_shape = tf.shape(lowres_img)[:2]  # (height,width)

    # Take Random Crops from LR Images
    lowres_width = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[1] - lowres_crop_size + 1, dtype=tf.int32
    )
    lowres_height = tf.random.uniform(
        shape=(), maxval=lowres_img_shape[0] - lowres_crop_size + 1, dtype=tf.int32
    )

    # Take Same Crops from LR Images in HR Images
    highres_width = lowres_width * scale
    highres_height = lowres_height * scale

    lowres_img_cropped = lowres_img[
        lowres_height : lowres_height + lowres_crop_size,
        lowres_width : lowres_width + lowres_crop_size,
    ]  # INPUT_DIM x IMPUT_DIM
    highres_img_cropped = highres_img[
        highres_height : highres_height + hr_crop_size,
        highres_width : highres_width + hr_crop_size,
    ]  # LABEL_SIZE x LABEL_SIZE

    return lowres_img_cropped, highres_img_cropped

def random_flip(lr_img, hr_img):
    flip_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(flip_chance < 0.5, lambda: (lr_img, hr_img), lambda: (tf.image.flip_left_right(lr_img), tf.image.flip_left_right(hr_img)))

def random_rotate(lr_img, hr_img):
    rotate_option = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    return tf.image.rot90(lr_img, rotate_option), tf.image.rot90(hr_img, rotate_option)

def random_lr_jpeg_noise(lr_img, hr_img, min_jpeg_quality=50, max_jpeg_quality=95):
    jpeg_noise_chance = tf.random.uniform(shape=(), maxval=1)
    return tf.cond(jpeg_noise_chance < 0.5, lambda: (lr_img, hr_img), lambda: (tf.image.random_jpeg_quality(lr_img, min_jpeg_quality, max_jpeg_quality), hr_img))

def resize_image(image_array, factor):
  original_image = Image.fromarray(image_array)
  new_size = np.array(original_image.size) * factor
  new_size = new_size.astype(np.int32)
  new_size = tuple(new_size)
  resized = original_image.resize(new_size, Image.Resampling.BICUBIC)
  resized = img_to_array(resized)
  resized = resized.astype(np.uint8)
  return resized

def downsize_upsize_image(image, scale):
  scaled = resize_image(image, 1.0/scale)
  scaled = resize_image(scaled, scale/1.0)
  return scaled

def tight_crop_image(image, scale):
  height, width = image.shape[:2]
  width -= int(width % scale)
  height -= int(height % scale)
  return image[:height, :width]

def crop_input(image, x, y, INPUT_DIM):
  y_slice = slice(y, y + INPUT_DIM)
  x_slice = slice(x, x + INPUT_DIM)
  return image[y_slice, x_slice]

def crop_output(image, x, y, PAD, LABEL_SIZE):
  y_slice = slice(int(y + PAD), int(y + PAD + LABEL_SIZE))
  x_slice = slice(int(x + PAD), int(x + PAD + LABEL_SIZE))
  return image[y_slice, x_slice]

def normalizer(sampleLR, sampleHR):
  sampleHR = tf.cast(sampleHR, tf.float32) / 255.
  sampleLR = tf.cast(sampleLR, tf.float32) / 255.
  return (sampleLR, sampleHR)

#endregion

# region - Visualization -
def plot_images(images):
  nImages = len(images)
  fig, ax = plt.subplots(ncols=nImages, figsize=(10 * nImages, 14), squeeze=True)
  for i in range(0, nImages):
    ax[i].imshow(images[i])
    ax[i].title.set_text(f"Image {i} | Size: {images[i].shape}")
    ax[i].axis("off")

def slice_images(images, LABEL_SIZE):
  # Create the Shape of the Output
  height, width = images[0].shape[:2]

  y = x = 0
  while 1:
      if(y <= height - LABEL_SIZE + 1): y+=LABEL_SIZE
      else: break;
  
  while 1:
      if(x <= width - LABEL_SIZE + 1): x+=LABEL_SIZE
      else: break;

  # Slice Images
  images_sliced = []
  for img in images:
    current = img[
      0:(y),
      0:(x),
      :
    ]
    images_sliced.append(current)
  return images_sliced

# endregion

# region - Metrics -
def PSNR_metric(y_true, y_pred):
  psnr = tf.image.psnr(y_true, y_pred, max_val=255)
  return psnr

def SSIM_metric(y_true, y_pred):
  ssim = tf.image.ssim(y_true, y_pred, max_val=255)
  return ssim

def MSE_metric(y_true,y_pred):
  y_true = np.array(y_true, np.float32)
  y_pred = np.array(y_pred, np.float32)
  return tf.reduce_mean( (y_true - y_pred) ** 2 )

# endregion

# region - Custom Predicts -
def predict_vdsr(model, image, PAD, LABEL_SIZE, SCALE):
    output = np.zeros(image.shape)
    height, width = output.shape[:2]
    # Residual
    for y in range(0, height - LABEL_SIZE + 1, LABEL_SIZE):
        for x in range(0, width - LABEL_SIZE + 1, LABEL_SIZE):
            crop = crop_output(image, x, y, PAD, LABEL_SIZE)
            scaled = resize_image(crop, 1.0/SCALE)
            image_batch = np.expand_dims(scaled, axis=0)
            prediction = model.predict(image_batch)
            output_y_slice = slice(y + PAD, y + PAD + LABEL_SIZE)
            output_x_slice = slice(x + PAD, x + PAD + LABEL_SIZE)
            output[output_y_slice, output_x_slice] = prediction
    return output
# endregion
