import os
import shutil
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import models.I_SRCNN.utilities.utils_srcnn as us

from glob import glob
from tf_keras.layers import *
from tf_keras.optimizers import *
from tf_keras.models import load_model
from tf_keras.preprocessing.image import *

from distutils.dir_util import copy_tree


def key_func(mf):
    return mf.score;

class Iperparameters():
    def __init__(self, scale, input_dim, label_size, stride, pad):
        self.scale      = scale
        self.input_dim  = input_dim
        self.label_size = label_size
        self.stride     = stride
        self.pad        = pad

class ModelFolder():
    model_type = ""
    model_name = ""
    model_path = ""
    log_path = ""

    def __init__(self, model_type, model_name, model_data_path, logs_data_path, iper):
        self.model_type = model_type
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

    def EvaluateModel(self, test_image):
        if(self.model_type == "vanilla"):
            scaled = us.downsize_upsize_image(test_image, self.iper.scale)
            self.img_pred = us.predict_vanilla(model=self.model, scaled=scaled, INPUT_DIM=self.iper.input_dim, PAD=self.iper.pad[0], LABEL_SIZE=self.iper.label_size[0])
        elif(self.model_type == "residual"):
            self.img_pred = us.predict_residual(model=self.model, image=test_image, PAD=self.iper.pad[1], LABEL_SIZE=self.iper.label_size[1], SCALE=self.iper.scale)
        self.img_pred = np.array(self.img_pred, np.uint8);

        img_pred_sliced, test_image_sliced = us.slice_images([self.img_pred, test_image])
        self.psnr = us.PSNR_metric(np.array(test_image_sliced, np.uint8), np.array(img_pred_sliced, np.uint8));
        self.ssim = us.SSIM_metric(np.array(test_image_sliced, np.uint8), np.array(img_pred_sliced, np.uint8));
        self.score = (self.psnr * 0.5 + self.ssim * 0.5) / 2.;
        # Round Results
        self.psnr  = round(self.psnr.numpy() * 1000) / 1000.
        self.ssim  = round(self.ssim.numpy() * 1000) / 1000.
        self.score = round(self.score.numpy() * 1000) / 1000.


class EvaluateAllModels():
    models_path = ""
    logs_path = ""
    test_image_path = ""
    model_folders = []

    def __init__(self, base_path, iper):
        self.models_path = base_path + "\\content\\models";
        self.logs_path = base_path + "\\content\\tensorboard";
        self.iper=iper
        self.FillModelsList();

    def FillModelsList(self):
        model_type_dir = (pathlib.Path(self.models_path) / "*")
        model_type_dir = str(model_type_dir)
        model_type_paths = [*glob(model_type_dir)]
        for model_type_path in model_type_paths:
            model_type = model_type_path.replace(str(self.models_path + "\\"), "")
            model_dir = (pathlib.Path(model_type_path) / "*")
            model_dir = str(model_dir)
            model_paths = [*glob(model_dir)]
            for model_data_path in model_paths:
                model_name = model_data_path.replace(str(model_type_path + "\\"), "")
                logs_data_path = self.logs_path + "\\archive\\" + model_name;
                self.model_folders.append(ModelFolder(model_type, model_name, model_data_path, logs_data_path, iper=self.iper))
                #print("=================================================================");
                #print("model_name      =", model_name)
                #print("model_data_path =", model_data_path)
                #print("logs_data_path  =", logs_data_path)

    def SetTestImage(self, image_path):
        self.test_image_path = image_path

    def EvaluateAllModels(self):
        test_image = load_img(self.test_image_path)
        test_image = img_to_array(test_image)
        test_image = test_image.astype(np.uint8)
        test_image = us.tight_crop_image(test_image, self.iper.scale)
        for mf in self.model_folders:
            mf.EvaluateModel(test_image)
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

        '''
        # Set the right-hand Y-axis ticks and labels
        ax2 = ax1.twinx()
        # Set equal limits on both yaxis so that the ticks line up
        ax2.set_ylim(ax1.get_ylim())
        # Set the tick locations and labels
        ax2.set_yticks(
            np.arange(len(self.model_folders)),
            labels=[format_score(score) for score in scores_by_test.values()])

        ax2.set_ylabel('Test Scores')
        '''

    def PredImagesInPerformanceOrder(self):
        test_image = load_img(self.test_image_path)
        test_image = img_to_array(test_image)
        test_image = test_image.astype(np.uint8)
        test_image = us.tight_crop_image(test_image, self.iper.scale)
        scaled = us.downsize_upsize_image(test_image, self.iper.scale)
        test_image_sliced, scaled_sliced = us.slice_images([test_image, scaled])

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

        i = j = 0
        for i in range(2):
            for j in range(2):
                ax[i + 1, j].imshow((np.array(us.slice_images([self.model_folders[i * 2 + j].img_pred])[0], np.uint8)))
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


