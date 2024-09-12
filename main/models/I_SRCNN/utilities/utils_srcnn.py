import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image
from tf_keras.preprocessing.image import img_to_array, load_img
from tf_keras.models import load_model

from skimage.metrics import structural_similarity as ssim

# - Data Augmentation -
def normalizer(sampleLR, sampleHR):
  sampleHR = tf.cast(sampleHR, tf.float32) / 255.
  sampleLR = tf.cast(sampleLR, tf.float32) / 255.
  return (sampleLR, sampleHR)

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

def change_model_name(model_path, new_name):
    model = load_model(filepath=model_path)
    model._name = new_name
    model.save(filepath=model_path)



# - Visualization -
def plot_images(images):
  nImages = len(images)
  fig, ax = plt.subplots(ncols=nImages, figsize=(10 * nImages, 14), squeeze=True)
  for i in range(0, nImages):
    ax[i].imshow(images[i])
    ax[i].title.set_text(f"Image {i} | Size: {images[i].shape}")
    ax[i].axis("off")

def plot_images_dict(images, maxCols=4):
    nImages = len(images)
    ncols = maxCols
    nrows = max(int(np.ceil(nImages/maxCols)), 2)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10), squeeze=True)
    i = j = 0
    for name in images:
      if(j == ncols):
        j = 0
        i += 1

      ax[i, j].imshow(images[name])
      psnr = PSNR_metric(images["Original"], images[name])
      psnr = round(float(psnr), 2)
      mse = MSE_metric(images["Original"], images[name])
      mse = round(float(mse), 2)
      #ax[i, j].title.set_text(f"{name} | psnr: {psnr}, mse: {mse}")
      ax[i, j].title.set_text(f"{name} | psnr: {psnr}")
      ax[i, j].axis("off")
      j+=1

def plot_images_dict_base(images, maxCols=4):
    nImages = len(images)
    ncols = maxCols
    nrows = max(int(np.ceil(nImages/maxCols)), 2)

    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10), squeeze=True)
    i = j = 0
    for name in images:
      if(j == ncols):
        j = 0
        i += 1

      ax[i, j].imshow(images[name])
      ax[i, j].title.set_text(f"{name}")
      ax[i, j].axis("off")
      j+=1

def slice_images(images, slicing_pad=15, multiply=2):
  # Create the Shape of the Output
  height, width = images[0].shape[:2]
  # Slice Images
  images_sliced = []
  for img in images:
    current = img[
      slicing_pad:(height - slicing_pad * multiply),
      slicing_pad:(width - slicing_pad * multiply),
      :
    ]
    images_sliced.append(current)
  return images_sliced



# - Metrics -
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


# - Predictions -
def predict_vanilla(model, scaled, INPUT_DIM, PAD, LABEL_SIZE):
    output = np.zeros(scaled.shape)
    height, width = output.shape[:2]
    # Vanilla
    for y in range(0, height - INPUT_DIM + 1, LABEL_SIZE):
        for x in range(0, width - INPUT_DIM + 1, LABEL_SIZE):
            crop = crop_input(scaled, x, y, INPUT_DIM)
            image_batch = np.expand_dims(crop, axis=0)
            prediction = model.predict(image_batch)
            output_y_slice = slice(y + PAD, y + PAD + LABEL_SIZE)
            output_x_slice = slice(x + PAD, x + PAD + LABEL_SIZE)
            output[output_y_slice, output_x_slice] = prediction
    return output

def predict_residual(model, image, PAD, LABEL_SIZE, SCALE):
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

'''
 - DEPRECATED -
def evaluate_vanilla(image_test_path, model, SCALE, INPUT_DIM, PAD, LABEL_SIZE):
  image = load_img(image_test_path)
  image = img_to_array(image)
  image = image.astype(np.uint8)
  image = tight_crop_image(image, SCALE)
  scaled = downsize_upsize_image(image, SCALE)

  output = predict_vanilla(model, scaled, INPUT_DIM, PAD, LABEL_SIZE)

  scaled = downsize_upsize_image(image, SCALE)
  images_sliced = slice_images([image, scaled])
  images_sliced.append(output)

  plot_images(np.array(images_sliced, np.uint8))
  PlotERRORs(np.array(images_sliced, np.float32))

def evaluate_residual(image_test_path, model, SCALE, INPUT_DIM, PAD, LABEL_SIZE):
  image = load_img(image_test_path)
  image = img_to_array(image)
  image = image.astype(np.uint8)
  image = tight_crop_image(image, SCALE)

  output = predict_residual(model, image, INPUT_DIM, PAD, LABEL_SIZE)

  scaled = downsize_upsize_image(image, SCALE)
  images_sliced = slice_images([image, scaled])
  images_sliced.append(output)

  plot_images(np.array(images_sliced, np.uint8))
  PlotERRORs(np.array(images_sliced, np.float32))

def predict_vanilla(model, scaled, INPUT_DIM, PAD, LABEL_SIZE):
    output = np.zeros(scaled.shape)
    height, width = output.shape[:2]
    # Vanilla
    for y in range(0, height - INPUT_DIM + 1, LABEL_SIZE):
        for x in range(0, width - INPUT_DIM + 1, LABEL_SIZE):
            crop = crop_input(scaled, x, y, INPUT_DIM)
            image_batch = np.expand_dims(crop, axis=0)
            prediction = model.predict(image_batch)
            output_y_slice = slice(y + PAD, y + PAD + LABEL_SIZE)
            output_x_slice = slice(x + PAD, x + PAD + LABEL_SIZE)
            output[output_y_slice, output_x_slice] = prediction
    return slice_images([output])[0]

def predict_residual(model, image, PAD, LABEL_SIZE, SCALE):
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
    return slice_images([output])[0]
'''