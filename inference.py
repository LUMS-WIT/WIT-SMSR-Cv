import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR
from utils import  SREncDataset
from utils import plot_sample, plot_predictions , plot_difference 

# === Configs ===
COARSE_DIR = "training/central_valley/test/9km"
FINE_DIR = "training/central_valley/test/1km"
MODEL_PATH = "checkpoints/central_valley/best_model_GeoTransformerSR_1km.pth"
BATCH_SIZE = 1
SCALE_FACTOR = 9
UPSAMPLE = False
SMAP=True
COARSE_RES = "9km"
FINE_RES = "1km"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GEOMODEL = True


# === Load dataset (only coarse is essential for inference) ===
dataset = SREncDataset(
    coarse_dir= COARSE_DIR,
    fine_dir= FINE_DIR,
    coarse_res=COARSE_RES,
    fine_res= FINE_RES, 
    scale_factor=SCALE_FACTOR,
    SMAP=SMAP, 
    upsample= UPSAMPLE
)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Load model ===
# model = SRCNN().to(DEVICE)
# model = TransformerSkip(scale_factor=SCALE_FACTOR).to(DEVICE)
model = GeoTransformerSR(scale_factor=SCALE_FACTOR).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# === Inference and plotting ===
with torch.no_grad():
    for idx, batch in enumerate(loader):
        # SREncDataset yields 6 fields
        if len(batch) == 2:
            coarse, fine = batch
            center_lon = center_lat = date_str = coarse_path = None  # Just placeholders
        else:
            coarse, fine, center_lon, center_lat, date_str, coarse_path = batch

        coarse = coarse.to(DEVICE)

        if GEOMODEL:
            pred = model(coarse, center_lat, center_lon, date_str)
        else:
            pred = model(coarse)
        pred = pred.clamp(0.0, 1.0)

        # plot_sample(fine.cpu(), pred.cpu(), title=f"Prediction Sample #{idx}")
        plot_predictions(
            coarse.cpu(), 
            fine.cpu(), 
            pred.cpu(), 
            coarse_res=COARSE_RES, 
            fine_res=FINE_RES, 
            title=f"Prediction Sample #{idx}"
        )

        plot_difference(
            fine.cpu(), 
            pred.cpu(), 
            fine_res="1km", 
            title=f"RMSE Sample # {idx}"
            )


        # Optional: break early if just testing
        if idx >= 1:
            break

