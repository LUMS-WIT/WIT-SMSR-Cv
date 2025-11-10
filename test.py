import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
from torch.utils.data import DataLoader
from model import SRCNN_Shuffle
from transformer import Transformer, TransformerSkip, GeoTransformerSR
from utils import compare_images, mean_r, LossLogger, SREncDataset
from tqdm import tqdm
import matplotlib.pyplot as plt



# === Configs ===
COARSE_DIR = "training/central_valley/SMAP-HB/test/9km"
FINE_DIR = "training/central_valley/SMAP-HB/test/1km"
MODEL_PATH = "checkpoints/central_valley/best_model.pth"
BATCH_SIZE = 1
SCALE_FACTOR = 9
UPSAMPLE = False         # True for SRCNN or basleline test
SMAP = True
COARSE_RES = "9km"
FINE_RES = "1km"
# SAVE_DIR = "checkpoints"

GEOMODEL = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(loader):


    # === Load model ===
    # model = SRCNN_Shuffle(scale_factor=SCALE_FACTOR).to(DEVICE)
    # model = Transformer(scale_factor=SCALE_FACTOR).to(DEVICE)
    # model = TransformerSkip(scale_factor=SCALE_FACTOR).to(DEVICE)
    model = GeoTransformerSR(scale_factor=SCALE_FACTOR).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE,  weights_only=True))
    model.eval()

    # === Evaluate ===
    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    with torch.no_grad():
        for batch_id, batch in enumerate(tqdm(loader, desc="Testing")):
            # Unpack batch according to loader; always expects 6 fields!
            # (coarse, fine, center_lon, center_lat, date_str, coarse_path)
            coarse, fine, center_lon, center_lat, date_str, coarse_path = batch
            
            # Move tensors to device
            coarse = coarse.to(DEVICE)
            fine = fine.to(DEVICE)

            # Predict
            if GEOMODEL:
                pred = model(coarse, center_lat, center_lon, date_str)
            else:
                pred = model(coarse)
            
            pred = pred.clamp(0.0, 1.0)

            # Convert to numpy
            pred_np = pred[0, 0].cpu().numpy()
            fine_np = fine[0, 0].cpu().numpy()

            # Clip and compute metrics
            pred_np = np.clip(pred_np, 0, 1)
            fine_np = np.clip(fine_np, 0, 1)

            metrics = compare_images(pred_np, fine_np)

            # print(f"[{batch_id}] PSNR: {metrics['PSNR']:.2f}, MSE: {metrics['MSE']:.4f}, "
            #       f"Bias: {metrics['Bias']:.4f}, ubRMSE: {metrics['ubRMSE']:.4f}")

            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    # === Aggregate ===
    print("\nBaseline on Test Set:")
    for k in all_metrics:
        if k == "PearsonR":
            mean_val = mean_r(all_metrics[k])
        else:
            mean_val = np.mean(all_metrics[k])
        print(f"{k}: {mean_val:.4f}")


def baseline():
    # === Load dataset ===
    dataset = SREncDataset(
        coarse_dir= COARSE_DIR,
        fine_dir= FINE_DIR,
        coarse_res=COARSE_RES,
        fine_res= FINE_RES, 
        scale_factor=SCALE_FACTOR,
        SMAP=SMAP, 
        upsample=True
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_metrics = {"PSNR": [], "MSE": [], "Bias": [], "ubRMSE": [], "PearsonR": []}

    for batch_id, batch in enumerate(tqdm(loader, desc="Testing")):
        # Support both (coarse, fine) and (coarse, fine, ...)
        if len(batch) == 2:
            coarse, fine = batch
        else:
            coarse, fine, *_ = batch

        coarse_np = coarse[0, 0].cpu().numpy()
        fine_np = fine[0, 0].cpu().numpy()
        coarse_np = np.clip(coarse_np, 0, 1)
        fine_np = np.clip(fine_np, 0, 1)

        metrics = compare_images(coarse_np, fine_np)
        for k in all_metrics:
            all_metrics[k].append(metrics[k])

    # === Aggregate ===
    print("\nBaseline on Test Set:")
    for k in all_metrics:
        if k == "PearsonR":
            mean_val = mean_r(all_metrics[k])
        else:
            mean_val = np.mean(all_metrics[k])
        print(f"{k}: {mean_val:.4f}")



def logs_display():

    logger = LossLogger("checkpoints/training_log.json")
    logs = logger.get_logs()
    epochs = range(1, len(logs["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, logs["train_loss"], label="Train Loss")
    plt.plot(epochs, logs["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.title("Train/Val Loss")

    plt.figure()
    plt.plot(epochs, logs["train_psnr"], label="Train PSNR")
    plt.plot(epochs, logs["val_psnr"], label="Val PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.title("Train/Val PSNR") 
    plt.show() 


if __name__ == "__main__":
    
    # === Load dataset ===
    dataset = SREncDataset(
        coarse_dir=COARSE_DIR,
        fine_dir=FINE_DIR,
        coarse_res=COARSE_RES,
        fine_res= FINE_RES, 
        scale_factor=SCALE_FACTOR,
        SMAP=SMAP,
        upsample=UPSAMPLE
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    main(loader)
    print('\n')
    baseline()
    logs_display()
