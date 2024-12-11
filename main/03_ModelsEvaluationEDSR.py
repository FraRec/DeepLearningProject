import os
from utilities.models import EvaluateAllModels, Iperparameters

# Define Iperparameters
iperparameters = Iperparameters(
    scale=4,
    input_dim=24,
    label_size=96
)

# Define Path
base_path = os.path.dirname(os.path.realpath(__file__))
base_path += "\\models\\III_EDSR"

# Test Images Paths
IMG_NAME_HORSE      = base_path + "\\data\\test\\horse_GT.bmp"

# - Eval Models -
evalModels = EvaluateAllModels(base_path=base_path, iper=iperparameters)
evalModels.SetTestImage(IMG_NAME_HORSE)
evalModels.EvaluateAllModels()
evalModels.PlotResults();
#evalModels.PlotTensorboard();

 