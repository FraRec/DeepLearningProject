import os
from models.I_SRCNN.utilities.model_srcnn import EvaluateAllModels, Iperparameters

# Define Iperparameters
INPUT_DIM   = 32
LABEL_SIZE  = [20, 64]
PAD         = [int((INPUT_DIM - LABEL_SIZE[0]) / 2.0), 0]
iperparameters = Iperparameters(
    scale=2.0,
    input_dim=32,
    label_size=[20, 64],
    stride=14,
    pad=[int((INPUT_DIM - LABEL_SIZE[0]) / 2.0), 0]
)

# Define Path
base_path = os.path.dirname(os.path.realpath(__file__))
base_path += "\\models\\I_SRCNN"

# Test Images Paths
IMG_NAME_BUTTERFLY  = base_path + "\\data\\test\\butterfly_GT.bmp"

# - Eval Models -
evalModels = EvaluateAllModels(base_path=base_path, iper=iperparameters)
evalModels.SetTestImage(IMG_NAME_BUTTERFLY)
evalModels.EvaluateAllModels()
evalModels.PlotResults();
#evalModels.PlotTensorboard();


