import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import utilities.utils as us

from glob import glob
from tf_keras.layers import *
from tf_keras.optimizers import *
from tf_keras.models import load_model
from tf_keras.preprocessing.image import *

from distutils.dir_util import copy_tree
import shutil


def key_func(mf):
    return mf.score;

class Iperparameters():
    def __init__(self, scale, input_dim, label_size, stride=0, pad=0):
        self.scale      = scale
        self.input_dim  = input_dim
        self.label_size = label_size
        self.stride     = stride,
        self.pad        = pad

class ModelFolder():

    def __init__(self, model_name, model_data_path, logs_data_path, iper):
        self.model_name = model_name
        self.iper = iper
        self.FillVariables(model_data_path, logs_data_path)
        self.model = load_model(self.model_path, compile=False)

    def FillVariables(self, model_data_path, logs_data_path):
        self.model_path = model_data_path + "\\model.h5"
        self.log_path = logs_data_path
        #print("=================================================================");
        #print("model_name =", self.model_name)
        #print("model_dir  =", self.model_path);
        #print("log_dir    =", self.log_path);

    def EvaluateModel(self, highres, lowres, predict_type="None"):
        if(predict_type == "None"):
            self.img_pred = self.model.predict(np.expand_dims(lowres, axis=0))[0]
        elif(predict_type == "VDSR"):
            self.img_pred = us.predict_vdsr(model=self.model, image=highres, PAD=self.iper.pad, LABEL_SIZE=self.iper.label_size, SCALE=self.iper.scale)

        img_pred_sliced, test_highres_sliced = us.slice_images([self.img_pred, highres], LABEL_SIZE=self.iper.label_size)
        
        self.psnr = us.PSNR_metric(np.array(test_highres_sliced, np.uint8), np.array(img_pred_sliced, np.uint8));
        self.ssim = us.SSIM_metric(np.array(test_highres_sliced, np.uint8), np.array(img_pred_sliced, np.uint8));
        self.score = (self.psnr * 0.5 + self.ssim * 0.5) / 2.;
        
        # Round Results
        self.psnr  = round(self.psnr.numpy() * 1000) / 1000.
        self.ssim  = round(self.ssim.numpy() * 1000) / 1000.
        self.score = round(self.score.numpy() * 1000) / 1000.

class EvaluateAllModels():
    
    def __init__(self, base_path, iper):
        self.models_path = base_path + "\\content\\models";
        self.logs_path = base_path + "\\content\\tensorboard";
        self.iper = iper
        self.model_folders =[]
        self.FillModelsList();

    def FillModelsList(self):
        model_dir = (pathlib.Path(self.models_path) / "*")
        model_dir = str(model_dir)
        model_paths = [*glob(model_dir)]
        for model_data_path in model_paths:
            model_name = model_data_path.replace(str(self.models_path + "\\"), "")
            logs_data_path = self.logs_path + "\\archive\\" + model_name;
            self.model_folders.append(ModelFolder(model_name, model_data_path, logs_data_path, self.iper))
            #print("=================================================================");
            #print("model_name      =", model_name)
            #print("model_data_path =", model_data_path)
            #print("logs_data_path  =", logs_data_path)

    def SetTestImage(self, image_path):
        self.test_image_path = image_path

    def EvaluateAllModels(self, time_label_size=5, predict_type="None"):
        test_image = load_img(self.test_image_path)
        test_image = img_to_array(test_image)
        test_image = test_image.astype(np.uint8)

        self.time_label_size = time_label_size
        image_size = self.iper.label_size * self.time_label_size
        halfHeight = test_image.shape[0] // 2
        halfWidth = test_image.shape[1] // 2
        halfImage = image_size // 2

        highres = us.crop_input(test_image, halfWidth - halfImage, halfHeight - halfImage, image_size)
        lowres = us.resize_image(highres, 1 / self.iper.scale)
        for mf in self.model_folders:
            mf.EvaluateModel(highres, lowres, predict_type)
        self.model_folders.sort(key=key_func, reverse=True)

    def VisualizeTensorboardData(self):
        # Specify the path of the directory to be deleted
        visualize_path = self.logs_path + "\\visualize";
        # Check if the directory exists before attempting to delete it
        if os.path.exists(visualize_path):
            shutil.rmtree(visualize_path)
            print(f"The directory {visualize_path} has been deleted.")
        else:
            print(f"The directory {visualize_path} does not exist.")
        os.mkdir(visualize_path)

        nImages = min(6, len(self.model_folders))
        #print("=================================================================");
        #print("visualize_path  =", visualize_path)
        for i in range(nImages):
            log_path = self.model_folders[i].log_path
            copy_tree(log_path, (visualize_path + "\\" + self.model_folders[i].model_name));
        os.system('tensorboard --logdir=' + visualize_path)

    def HorizontalBarPerformanceGraph(self):
        plt.style.use("fivethirtyeight");
        fig, ax1 = plt.subplots(figsize=(9, 7), layout='constrained')
        fig.canvas.manager.set_window_title('Performances')

        ax1.set_title("Higher is Better")
        ax1.set_xlabel("Score")

        model_folder_reversed = self.model_folders.copy();
        model_folder_reversed.reverse();
        test_names = [mf.model_name for mf in model_folder_reversed]
        scores = [mf.score for mf in model_folder_reversed]
        psnrs = [mf.psnr for mf in model_folder_reversed]
        ssims = [mf.ssim for mf in model_folder_reversed]
        y_indexes = np.arange(len(test_names))

        height = 0.25;
        rectsMax = ax1.barh(y_indexes + height,  psnrs, color="#444444", height=height, label="psnr")
        rectsMed = ax1.barh(y_indexes         , scores, color="#008fd5", height=height, label="Score")
        rectsMin = ax1.barh(y_indexes - height,  ssims, color="#e5ae38", height=height, label="ssim")

        #rectsMed = ax1.barh(y_indexes, scores, width=width, color="008fd5",  align='center', height=0.5)

        # Partition the percentile values to be able to draw large numbers in
        # white within the bar, and small numbers in black outside the bar.
        ax1.bar_label(rectsMax,  psnrs, padding=5, color='black', fontweight='bold')
        ax1.bar_label(rectsMed, scores, padding=-45, color='white', fontweight='bold')
        ax1.bar_label(rectsMin,  ssims, padding=5 , color='black', fontweight='bold')

        ax1.legend()
        ax1.set_yticks(ticks=y_indexes, labels=test_names);
        ax1.set_xlim([0, 35])
        ax1.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 35])
        ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
        ax1.axvline(50, color='grey', alpha=0.25)  # median position

    def PredImagesInPerformanceOrder(self):
        test_image = load_img(self.test_image_path)
        test_image = img_to_array(test_image)
        test_image = test_image.astype(np.uint8)

        image_size = self.iper.label_size * self.time_label_size
        halfHeight = test_image.shape[0] // 2
        halfWidth = test_image.shape[1] // 2
        halfImage = image_size // 2

        test_image_sliced = us.crop_input(test_image, halfWidth - halfImage, halfHeight - halfImage, image_size)
        scaled_sliced = us.resize_image(test_image_sliced, 1 / self.iper.scale)

        nImages = min(6, len(self.model_folders))
        nrows = max(int(np.ceil(nImages/2)), 2)
        ncols = 2
        
        fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(10, 10), squeeze=True)
        ax[0, 0].imshow(test_image_sliced)
        ax[0, 0].title.set_text("Original")
        ax[0, 0].axis("off")
        ax[0, 1].imshow(scaled_sliced)
        ax[0, 1].title.set_text("Bicubic")
        ax[0, 1].axis("off")

        '''
        ax[1, 0].imshow((np.array(us.slice_images([self.model_folders[0].img_pred], LABEL_SIZE=self.iper.label_size)[0], np.uint8)))
        name = self.model_folders[0].model_name
        score = self.model_folders[0].score
        ax[1, 0].title.set_text(f"{name} | score: {score}")
        ax[1, 0].axis("off")
        '''
        
        i = j = 0
        for i in range(2):
            for j in range(2):
                if(len(self.model_folders) <= i * 2 + j): continue;
                ax[i + 1, j].imshow((np.array(us.slice_images([self.model_folders[i * 2 + j].img_pred], LABEL_SIZE=self.iper.label_size)[0], np.uint8)))
                name = self.model_folders[i * 2 + j].model_name
                score = self.model_folders[i * 2 + j].score
                ax[i + 1, j].title.set_text(f"{name} | score: {score}")
                ax[i + 1, j].axis("off")

    def PlotResults(self):
        self.HorizontalBarPerformanceGraph();
        self.PredImagesInPerformanceOrder();
        plt.show();
        return;

    def PlotTensorboard(self):
        self.VisualizeTensorboardData()
        return;



