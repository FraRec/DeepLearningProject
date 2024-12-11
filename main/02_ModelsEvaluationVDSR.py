import os
from utilities.models import EvaluateAllModels, Iperparameters

# Define Iperparameters
iperparameters = Iperparameters(
    scale=2.0,
    input_dim=32,
    label_size=64,
    stride=14,
    pad=0
)

# Define Path
base_path = os.path.dirname(os.path.realpath(__file__))
base_path += "\\models\\II_VDSR"

# Test Images Paths
IMG_NAME_BUTTERFLY  = base_path + "\\data\\test\\butterfly_GT.bmp"
IMG_NAME_BIRD       = base_path + "\\data\\test\\bird_GT.bmp"
IMG_NAME_WOMAN      = base_path + "\\data\\test\\woman_GT.bmp"

# - Eval Models -
evalModels = EvaluateAllModels(base_path=base_path, iper=iperparameters)
evalModels.SetTestImage(IMG_NAME_BUTTERFLY)
evalModels.EvaluateAllModels(time_label_size=4, predict_type="VDSR")
evalModels.PlotResults();
#evalModels.PlotTensorboard();

