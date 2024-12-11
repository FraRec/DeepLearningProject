import os
from utilities.models import EvaluateAllModels, Iperparameters

# Define Iperparameters
iperparameters = Iperparameters(
    scale=2,
    input_dim=32,
    label_size=64
)

# Define Path
base_path = os.path.dirname(os.path.realpath(__file__))
base_path += "\\models\\IV_RDN"

# Test Images Paths
IMG_NAME_HORSE      = base_path + "\\data\\test\\horse_GT.bmp"

# - Eval Models -
evalModels = EvaluateAllModels(base_path=base_path, iper=iperparameters)
evalModels.SetTestImage(IMG_NAME_HORSE)
evalModels.EvaluateAllModels(time_label_size=8)
evalModels.PlotResults();
#evalModels.PlotTensorboard();


 